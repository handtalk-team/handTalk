/**
 * handTalk Smart Glove Firmware
 * ==============================
 * Target  : ESP32 (ESP-IDF v5.x + NimBLE stack)
 * Hardware: 5× Flex sensors (ADC), 1× IMU (MPU-6050 via I2C)
 *
 * BLE GATT structure
 * ------------------
 * Service UUID : 4fafc201-1fb5-459e-8fcc-c5c9c331914b
 * Characteristic: beb5483e-36e1-4688-b7f5-ea07361b26a8
 *   Properties  : NOTIFY
 *   Value (28 bytes, little-endian):
 *     [0..1]   seq       uint16  — monotonic packet counter
 *     [2..6]   flex[5]   uint8   — ADC reading mapped to 0-255
 *     [7..12]  accel[3]  int16   — raw accel * 1000 (m/s²)
 *     [13..18] gyro[3]   int16   — raw gyro  * 1000 (rad/s)
 *     [19..26] quat[4]   int16   — quaternion * 10000 (w,x,y,z)
 *     [27]     quality   uint8   — RSSI-derived quality 0-255
 *
 * Pin mapping (adjust to actual PCB layout)
 * ------------------------------------------
 * Flex sensors: GPIO34(thumb), GPIO35(index), GPIO36(middle),
 *               GPIO39(ring), GPIO32(pinky)   — ADC1 channels
 * IMU SDA      : GPIO21
 * IMU SCL      : GPIO22
 * LED status   : GPIO2 (built-in)
 *
 * Build
 * -----
 *   cd firmware/esp32
 *   idf.py set-target esp32
 *   idf.py build flash monitor
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "driver/adc.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"

/* NimBLE */
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char *TAG = "handTalk";

/* ─── Pin / ADC configuration ──────────────────────────────── */
#define FLEX_COUNT       5
#define FLEX_ADC_UNIT    ADC_UNIT_1
static const adc_channel_t FLEX_CHANNELS[FLEX_COUNT] = {
    ADC_CHANNEL_6,  /* GPIO34 — thumb  */
    ADC_CHANNEL_7,  /* GPIO35 — index  */
    ADC_CHANNEL_0,  /* GPIO36 — middle */
    ADC_CHANNEL_3,  /* GPIO39 — ring   */
    ADC_CHANNEL_4,  /* GPIO32 — pinky  */
};
/* Calibration: map ADC counts to 0-255 (updated by calibration task) */
static int flex_min[FLEX_COUNT] = {1500, 1500, 1500, 1500, 1500};
static int flex_max[FLEX_COUNT] = {3500, 3500, 3500, 3500, 3500};

/* ─── I2C / MPU-6050 ────────────────────────────────────────── */
#define I2C_PORT         I2C_NUM_0
#define I2C_SDA_PIN      21
#define I2C_SCL_PIN      22
#define I2C_FREQ_HZ      400000
#define MPU6050_ADDR     0x68
#define MPU6050_PWR_MGMT 0x6B
#define MPU6050_ACCEL_H  0x3B
#define MPU6050_GYRO_H   0x43

/* Raw sensor data structure shared between sensor task and BLE task */
typedef struct {
    uint16_t seq;
    uint8_t  flex[FLEX_COUNT];   /* 0-255 */
    int16_t  accel[3];           /* ×1000 m/s² */
    int16_t  gyro[3];            /* ×1000 rad/s */
    int16_t  quat[4];            /* ×10000 */
    uint8_t  quality;            /* BLE RSSI quality 0-255 */
} __attribute__((packed)) sensor_packet_t;

static sensor_packet_t g_packet;
static SemaphoreHandle_t g_mutex;

/* ─── BLE UUIDs ─────────────────────────────────────────────── */
static const ble_uuid128_t GATT_SVC_UUID = BLE_UUID128_INIT(
    0x4f, 0xaf, 0xc2, 0x01, 0x1f, 0xb5, 0x45, 0x9e,
    0x8f, 0xcc, 0xc5, 0xc9, 0xc3, 0x31, 0x91, 0x4b
);
static const ble_uuid128_t GATT_CHR_UUID = BLE_UUID128_INIT(
    0xbe, 0xb5, 0x48, 0x3e, 0x36, 0xe1, 0x46, 0x88,
    0xb7, 0xf5, 0xea, 0x07, 0x36, 0x1b, 0x26, 0xa8
);

static uint16_t g_conn_handle  = BLE_HS_CONN_HANDLE_NONE;
static uint16_t g_chr_val_hdl  = 0;
static uint16_t g_notify_hdl   = 0;

/* ─── I2C helpers ───────────────────────────────────────────── */
static void i2c_init(void) {
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = I2C_SDA_PIN,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_io_num       = I2C_SCL_PIN,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ_HZ,
    };
    i2c_param_config(I2C_PORT, &conf);
    i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0);
}

static void mpu6050_write(uint8_t reg, uint8_t val) {
    uint8_t buf[2] = { reg, val };
    i2c_master_write_to_device(I2C_PORT, MPU6050_ADDR, buf, 2, pdMS_TO_TICKS(10));
}

static void mpu6050_read(uint8_t reg, uint8_t *out, size_t len) {
    i2c_master_write_read_device(
        I2C_PORT, MPU6050_ADDR, &reg, 1, out, len, pdMS_TO_TICKS(10));
}

static void mpu6050_init(void) {
    mpu6050_write(MPU6050_PWR_MGMT, 0x00);  /* wake up */
    vTaskDelay(pdMS_TO_TICKS(100));
}

/* ─── ADC helpers ───────────────────────────────────────────── */
static void adc_init(void) {
    adc1_config_width(ADC_WIDTH_BIT_12);
    for (int i = 0; i < FLEX_COUNT; i++) {
        adc1_config_channel_atten(FLEX_CHANNELS[i], ADC_ATTEN_DB_11);
    }
}

static uint8_t read_flex(int idx) {
    int raw = adc1_get_raw(FLEX_CHANNELS[idx]);
    int mapped = (raw - flex_min[idx]) * 255 / (flex_max[idx] - flex_min[idx]);
    if (mapped < 0)   mapped = 0;
    if (mapped > 255) mapped = 255;
    return (uint8_t)mapped;
}

/* ─── Sensor task (50 Hz) ───────────────────────────────────── */
static void sensor_task(void *arg) {
    mpu6050_init();
    uint16_t seq = 0;

    while (1) {
        /* Read flex sensors */
        uint8_t flex_raw[FLEX_COUNT];
        for (int i = 0; i < FLEX_COUNT; i++) {
            flex_raw[i] = read_flex(i);
        }

        /* Read IMU (6 bytes accel + 6 bytes gyro = 12 bytes starting at 0x3B) */
        uint8_t imu_buf[14];
        mpu6050_read(MPU6050_ACCEL_H, imu_buf, 14);

        /* Accel: raw int16 × 9.81 / 16384 (±2g range) → m/s² × 1000 */
        int16_t ax_raw = (int16_t)((imu_buf[0] << 8) | imu_buf[1]);
        int16_t ay_raw = (int16_t)((imu_buf[2] << 8) | imu_buf[3]);
        int16_t az_raw = (int16_t)((imu_buf[4] << 8) | imu_buf[5]);

        /* Gyro: raw int16 / 131 → deg/s → rad/s × 1000 */
        int16_t gx_raw = (int16_t)((imu_buf[8]  << 8) | imu_buf[9]);
        int16_t gy_raw = (int16_t)((imu_buf[10] << 8) | imu_buf[11]);
        int16_t gz_raw = (int16_t)((imu_buf[12] << 8) | imu_buf[13]);

        float ax = ax_raw * 9.81f / 16384.0f;
        float ay = ay_raw * 9.81f / 16384.0f;
        float az = az_raw * 9.81f / 16384.0f;

        float gx = gx_raw / 131.0f * (float)M_PI / 180.0f;
        float gy = gy_raw / 131.0f * (float)M_PI / 180.0f;
        float gz = gz_raw / 131.0f * (float)M_PI / 180.0f;

        /*
         * Quaternion from accelerometer (gravity-only, no Madgwick yet).
         * TODO: integrate Madgwick / Mahony filter for full orientation.
         */
        float norm = sqrtf(ax*ax + ay*ay + az*az);
        float nx = ax/norm, ny = ay/norm, nz = az/norm;
        float qw = sqrtf((nz + 1.0f) / 2.0f);
        float qx = ny / (2.0f * qw + 1e-9f);
        float qy = -nx / (2.0f * qw + 1e-9f);
        float qz = 0.0f;
        /* normalise quaternion */
        float qn = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= qn; qx /= qn; qy /= qn; qz /= qn;

        xSemaphoreTake(g_mutex, portMAX_DELAY);
        g_packet.seq = seq++;
        for (int i = 0; i < FLEX_COUNT; i++) g_packet.flex[i] = flex_raw[i];
        g_packet.accel[0] = (int16_t)(ax * 1000);
        g_packet.accel[1] = (int16_t)(ay * 1000);
        g_packet.accel[2] = (int16_t)(az * 1000);
        g_packet.gyro[0]  = (int16_t)(gx * 1000);
        g_packet.gyro[1]  = (int16_t)(gy * 1000);
        g_packet.gyro[2]  = (int16_t)(gz * 1000);
        g_packet.quat[0]  = (int16_t)(qw * 10000);
        g_packet.quat[1]  = (int16_t)(qx * 10000);
        g_packet.quat[2]  = (int16_t)(qy * 10000);
        g_packet.quat[3]  = (int16_t)(qz * 10000);
        g_packet.quality  = 200;   /* updated by BLE RSSI monitor */
        xSemaphoreGive(g_mutex);

        /* Notify BLE central if connected */
        if (g_conn_handle != BLE_HS_CONN_HANDLE_NONE && g_notify_hdl != 0) {
            struct os_mbuf *om = ble_hs_mbuf_from_flat(&g_packet, sizeof(g_packet));
            if (om) {
                ble_gatts_notify_custom(g_conn_handle, g_notify_hdl, om);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(20));  /* 50 Hz */
    }
}

/* ─── BLE GATT characteristic handler ──────────────────────── */
static int gatt_chr_access(
    uint16_t conn_handle, uint16_t attr_handle,
    struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        xSemaphoreTake(g_mutex, portMAX_DELAY);
        os_mbuf_append(ctxt->om, &g_packet, sizeof(g_packet));
        xSemaphoreGive(g_mutex);
    }
    return 0;
}

static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &GATT_SVC_UUID.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid       = &GATT_CHR_UUID.u,
                .access_cb  = gatt_chr_access,
                .flags      = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &g_chr_val_hdl,
            },
            { 0 }
        },
    },
    { 0 }
};

/* ─── GAP event handler ─────────────────────────────────────── */
static int gap_event(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            g_conn_handle = event->connect.conn_handle;
            ESP_LOGI(TAG, "BLE connected  handle=%d", g_conn_handle);
            gpio_set_level(2, 1);  /* LED on */
        } else {
            g_conn_handle = BLE_HS_CONN_HANDLE_NONE;
            ble_app_advertise();
        }
        break;
    case BLE_GAP_EVENT_DISCONNECT:
        g_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        ESP_LOGI(TAG, "BLE disconnected");
        gpio_set_level(2, 0);
        ble_app_advertise();
        break;
    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == g_chr_val_hdl) {
            g_notify_hdl = event->subscribe.attr_handle;
            ESP_LOGI(TAG, "Notify subscribed");
        }
        break;
    default: break;
    }
    return 0;
}

static void ble_app_advertise(void) {
    struct ble_gap_adv_params adv_params = { .conn_mode = BLE_GAP_CONN_MODE_UND,
                                              .disc_mode = BLE_GAP_DISC_MODE_GEN };
    struct ble_hs_adv_fields fields = {
        .flags             = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP,
        .tx_pwr_lvl_is_present = 1,
        .tx_pwr_lvl        = BLE_HS_ADV_TX_PWR_LVL_AUTO,
        .name              = (uint8_t *)"HandTalk-Glove",
        .name_len          = 14,
        .name_is_complete  = 1,
    };
    ble_gap_adv_set_fields(&fields);
    ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER,
                      &adv_params, gap_event, NULL);
    ESP_LOGI(TAG, "Advertising as 'HandTalk-Glove'");
}

static void ble_app_on_sync(void) {
    ble_hs_id_infer_auto(0, NULL);
    ble_app_advertise();
}

static void nimble_host_task(void *arg) {
    nimble_port_run();
    nimble_port_freertos_deinit();
}

/* ─── app_main ──────────────────────────────────────────────── */
void app_main(void) {
    nvs_flash_init();
    g_mutex = xSemaphoreCreateMutex();

    /* Status LED */
    gpio_set_direction(2, GPIO_MODE_OUTPUT);

    /* Init peripherals */
    i2c_init();
    adc_init();

    /* NimBLE */
    nimble_port_init();
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_gatts_count_cfg(gatt_svcs);
    ble_gatts_add_svcs(gatt_svcs);
    ble_hs_cfg.sync_cb = ble_app_on_sync;
    nimble_port_freertos_init(nimble_host_task);

    /* Sensor task at 50 Hz */
    xTaskCreate(sensor_task, "sensor", 4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "handTalk glove firmware started");
}

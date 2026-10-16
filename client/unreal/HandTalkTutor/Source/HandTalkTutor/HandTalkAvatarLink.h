// handTalk — 서버 /ws/avatar 에 붙어, 서버가 실시간으로 만든 수어 동작({"type":"motion"})을 메타휴먼 뼈대에 바로 입힌다.
// 애니메이션 클립은 쓰지 않는다. 동작은 서버의 동작 생성 모델이 문장마다 새로 만든다.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "HandTalkAvatarLink.generated.h"

class IWebSocket;
class USkeletalMeshComponent;
struct FHandTalkLivePlayer;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnTutorReply, const FString&, Text, const TArray<FString>&, Gloss);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnRecognition, const FString&, Text, float, Confidence);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnMotionStarted, float, Duration);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnLinkStateChanged, bool, bConnected);

UCLASS(ClassGroup = (HandTalk), meta = (BlueprintSpawnableComponent))
class HANDTALKTUTOR_API UHandTalkAvatarLink : public UActorComponent
{
	GENERATED_BODY()

public:
	UHandTalkAvatarLink();

	/** handTalk 서버의 아바타 전용 WebSocket 주소 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	FString ServerUrl = TEXT("ws://127.0.0.1:8000/ws/avatar");

	/** 끊기면 이 간격(초)으로 다시 연결 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	float ReconnectInterval = 3.0f;

	/** 움직일 메타휴먼 액터. 비우면 이 컴포넌트의 Owner. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TObjectPtr<AActor> TargetActor;

	/** 동작을 입힐 스켈레탈 메시 컴포넌트 이름 (몸과 옷). 각 메시가 자기 참조 포즈 기준으로 같은 동작을 받는다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TArray<FName> PartNames = { TEXT("Body"), TEXT("Torso"), TEXT("Legs"), TEXT("Feet") };

	/** 동작이 바뀔 때 섞는 시간(초) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	float BlendTime = 0.25f;

	/** Play 시작 시 화면을 이 카메라로 (Pixel Streaming 으로 보낼 시점) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TObjectPtr<AActor> ViewCamera;

	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnTutorReply OnTutorReply;
	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnRecognition OnRecognition;
	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnMotionStarted OnMotionStarted;
	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnLinkStateChanged OnLinkStateChanged;

	UFUNCTION(BlueprintCallable, Category = "HandTalk") void Connect();
	UFUNCTION(BlueprintCallable, Category = "HandTalk") void Disconnect();
	UFUNCTION(BlueprintPure, Category = "HandTalk") bool IsConnected() const;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	void SetupParts();
	void HandleMessage(const FString& Message);
	void HandleMotion(const TSharedPtr<class FJsonObject>& Root);
	void ScheduleReconnect();

	TSharedPtr<IWebSocket> Socket;
	bool bWantConnected = false;
	FTimerHandle ReconnectTimer;
	TArray<TWeakObjectPtr<USkeletalMeshComponent>> Parts;
	TSharedPtr<FHandTalkLivePlayer> LivePlayer;
};

// handTalk — 서버 /ws/avatar 에 붙어 튜터 응답(avatar_commands)과 학습 단계(lesson)를 받아 수어 클립을 실시간 재생한다.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "HandTalkAvatarLink.generated.h"

class IWebSocket;
class UAnimSequenceBase;
class USkeletalMeshComponent;

/** 서버 AvatarCommand (server/app/models/schemas/ws_messages.py) 와 같은 필드 */
USTRUCT(BlueprintType)
struct FHandTalkAvatarCommand
{
	GENERATED_BODY()

	/** 글로스 클립 이름. lexicon 의 clip 과 같고, Content 의 AnimSequence 이름과도 같다. */
	UPROPERTY(BlueprintReadWrite, EditAnywhere) FName Clip;
	UPROPERTY(BlueprintReadWrite, EditAnywhere) float BlendIn = 0.1f;
	UPROPERTY(BlueprintReadWrite, EditAnywhere) float BlendOut = 0.1f;
	UPROPERTY(BlueprintReadWrite, EditAnywhere) float Speed = 1.0f;
	UPROPERTY(BlueprintReadWrite, EditAnywhere) FString Expression;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnTutorReply, const FString&, Text, const TArray<FHandTalkAvatarCommand>&, Commands);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnRecognition, const FString&, Text, float, Confidence);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnClipStarted, FName, Clip);
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

	/** 클립을 재생할 메타휴먼 액터. 비우면 이 컴포넌트의 Owner. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TObjectPtr<AActor> TargetActor;

	/**
	 * 부위(스켈레탈 메시 컴포넌트 이름) → 클립 폴더. 옷 메시가 스켈레톤이 다른 메타휴먼(Vivian)은
	 * 부위마다 같은 이름의 클립을 따로 만든다 (Scripts/build_vivian_signs.py).
	 * 비어 있으면 Body 하나에 ClipFolder 를 쓴다.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TMap<FName, FString> PartClipFolders;

	/** PartClipFolders 가 비었을 때 Body 에 쓰는 폴더. Clip 이 KSL_hello 면 {ClipFolder}/KSL_hello */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	FString ClipFolder = TEXT("/Game/HandTalk/Signs");

	/** 쉬는 자세: 이 클립의 마지막 프레임을 잡고 있는다 (없으면 마지막 재생 자세 유지) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	FName RestClip = TEXT("KSL_hospital");

	/** 학습 단계의 목표 동작을 쉬는 동안 반복할 때의 쉬는 시간(초). 0 이면 반복 안 함 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	float DemoInterval = 1.8f;

	/** Play 시작 시 화면을 이 카메라로 (Pixel Streaming 으로 보낼 시점) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TObjectPtr<AActor> ViewCamera;

	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnTutorReply OnTutorReply;
	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnRecognition OnRecognition;
	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnClipStarted OnClipStarted;
	UPROPERTY(BlueprintAssignable, Category = "HandTalk") FOnLinkStateChanged OnLinkStateChanged;

	UFUNCTION(BlueprintCallable, Category = "HandTalk") void Connect();
	UFUNCTION(BlueprintCallable, Category = "HandTalk") void Disconnect();
	UFUNCTION(BlueprintPure, Category = "HandTalk") bool IsConnected() const;

	/** 서버 없이 테스트할 때: 명령 목록을 바로 재생 큐에 넣는다. */
	UFUNCTION(BlueprintCallable, Category = "HandTalk")
	void PlayCommands(const TArray<FHandTalkAvatarCommand>& Commands);

	/** 서버 없이 테스트할 때: 클립 이름 하나를 바로 재생한다. */
	UFUNCTION(BlueprintCallable, Category = "HandTalk")
	void PlayClipByName(FName Clip);

	/** 쉬는 동안 반복할 시범 클립 */
	UFUNCTION(BlueprintCallable, Category = "HandTalk")
	void SetDemoClips(const TArray<FName>& Clips);

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	struct FPart
	{
		TWeakObjectPtr<USkeletalMeshComponent> Mesh;
		FString Folder;
	};

	void SetupParts();
	void HandleMessage(const FString& Message);
	void ScheduleReconnect();
	void PlayNext();
	void Rest();
	void DemoTick();
	bool PlayOnParts(FName Clip, float Speed, float& OutLength);
	UAnimSequenceBase* ResolveClip(const FString& Folder, FName Clip) const;

	TSharedPtr<IWebSocket> Socket;
	bool bWantConnected = false;
	bool bPlaying = false;
	FTimerHandle ReconnectTimer;
	FTimerHandle ClipTimer;
	FTimerHandle DemoTimer;
	TArray<FHandTalkAvatarCommand> Queue;
	TArray<FName> DemoClips;
	TArray<FPart> Parts;
};

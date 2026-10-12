// handTalk — 서버 /ws/avatar 에 붙어 튜터 응답(avatar_commands)을 받아 수어 클립을 재생한다.

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

	/** 클립을 재생할 메타휴먼 Body 메시. 비워 두면 Owner 의 이름이 "Body" 인 컴포넌트를 찾는다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TObjectPtr<USkeletalMeshComponent> TargetMesh;

	/** 클립을 찾을 폴더. Clip 이름이 KSL_hello 이면 {ClipFolder}/KSL_hello 를 불러온다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	FString ClipFolder = TEXT("/Game/HandTalk/Signs");

	/** 클립이 없거나 큐가 비었을 때 재생할 대기 동작 (선택) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TObjectPtr<UAnimSequenceBase> IdleAnim;

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

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	void HandleMessage(const FString& Message);
	void ScheduleReconnect();
	void PlayNext();
	void PlayIdle();
	UAnimSequenceBase* ResolveClip(FName Clip) const;

	TSharedPtr<IWebSocket> Socket;
	bool bWantConnected = false;
	FTimerHandle ReconnectTimer;
	FTimerHandle ClipTimer;
	TArray<FHandTalkAvatarCommand> Queue;
};

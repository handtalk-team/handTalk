// handTalk — 서버 /ws/avatar 에 붙어, 서버가 실시간으로 만든 수어 동작을 메타휴먼 뼈대에 바로 입힌다.

#include "HandTalkAvatarLink.h"
#include "HandTalkLiveAnimInstance.h"

#include "Components/SkeletalMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "IWebSocket.h"
#include "Kismet/GameplayStatics.h"
#include "Modules/ModuleManager.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "TimerManager.h"
#include "WebSocketsModule.h"

DEFINE_LOG_CATEGORY_STATIC(LogHandTalk, Log, All);

UHandTalkAvatarLink::UHandTalkAvatarLink()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UHandTalkAvatarLink::BeginPlay()
{
	Super::BeginPlay();
	SetupParts();

	if (ViewCamera)
	{
		if (APlayerController* PC = UGameplayStatics::GetPlayerController(this, 0))
		{
			PC->SetViewTarget(ViewCamera);
		}
	}

	Connect();
}

void UHandTalkAvatarLink::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Disconnect();
	Super::EndPlay(EndPlayReason);
}

void UHandTalkAvatarLink::SetupParts()
{
	Parts.Reset();
	LivePlayer = MakeShared<FHandTalkLivePlayer>();
	LivePlayer->BlendTime = BlendTime;

	AActor* Target = TargetActor ? TargetActor.Get() : GetOwner();
	if (!Target)
	{
		return;
	}
	TArray<USkeletalMeshComponent*> Meshes;
	Target->GetComponents(Meshes);
	for (USkeletalMeshComponent* Mesh : Meshes)
	{
		if (!PartNames.Contains(Mesh->GetFName()))
		{
			continue;
		}
		// 리타깃/후처리 AnimBP 대신 받은 동작을 직접 입힌다 (옷 메시도 각자 같은 동작을 받으므로 Leader Pose 해제)
		Mesh->SetLeaderPoseComponent(nullptr);
		if (Mesh->GetFName() == TEXT("Body"))
		{
			Mesh->SetDisablePostProcessBlueprint(true);
		}
		Mesh->SetAnimInstanceClass(UHandTalkLiveAnimInstance::StaticClass());
		if (UHandTalkLiveAnimInstance* Inst = Cast<UHandTalkLiveAnimInstance>(Mesh->GetAnimInstance()))
		{
			Inst->Player = LivePlayer;
		}
		Parts.Add(Mesh);
	}
	UE_LOG(LogHandTalk, Log, TEXT("live motion parts: %d on %s"), Parts.Num(), *Target->GetName());
}

void UHandTalkAvatarLink::Connect()
{
	bWantConnected = true;
	if (Socket.IsValid() && Socket->IsConnected())
	{
		return;
	}

	FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
	Socket = FWebSocketsModule::Get().CreateWebSocket(ServerUrl);

	TWeakObjectPtr<UHandTalkAvatarLink> WeakThis(this);
	Socket->OnConnected().AddLambda([WeakThis]()
	{
		if (!WeakThis.IsValid()) return;
		UE_LOG(LogHandTalk, Log, TEXT("connected %s"), *WeakThis->ServerUrl);
		WeakThis->OnLinkStateChanged.Broadcast(true);
	});
	Socket->OnConnectionError().AddLambda([WeakThis](const FString& Error)
	{
		if (!WeakThis.IsValid()) return;
		UE_LOG(LogHandTalk, Warning, TEXT("connection error: %s"), *Error);
		WeakThis->ScheduleReconnect();
	});
	Socket->OnClosed().AddLambda([WeakThis](int32 Code, const FString& Reason, bool bClean)
	{
		if (!WeakThis.IsValid()) return;
		WeakThis->OnLinkStateChanged.Broadcast(false);
		WeakThis->ScheduleReconnect();
	});
	Socket->OnMessage().AddLambda([WeakThis](const FString& Message)
	{
		if (WeakThis.IsValid()) WeakThis->HandleMessage(Message);
	});

	Socket->Connect();
}

void UHandTalkAvatarLink::Disconnect()
{
	bWantConnected = false;
	if (UWorld* World = GetWorld())
	{
		World->GetTimerManager().ClearTimer(ReconnectTimer);
	}
	if (Socket.IsValid())
	{
		Socket->OnConnected().Clear();
		Socket->OnConnectionError().Clear();
		Socket->OnClosed().Clear();
		Socket->OnMessage().Clear();
		if (Socket->IsConnected())
		{
			Socket->Close();
		}
		Socket.Reset();
	}
}

bool UHandTalkAvatarLink::IsConnected() const
{
	return Socket.IsValid() && Socket->IsConnected();
}

void UHandTalkAvatarLink::ScheduleReconnect()
{
	UWorld* World = GetWorld();
	if (!bWantConnected || !World || World->GetTimerManager().IsTimerActive(ReconnectTimer))
	{
		return;
	}
	World->GetTimerManager().SetTimer(ReconnectTimer, this, &UHandTalkAvatarLink::Connect, ReconnectInterval, false);
}

void UHandTalkAvatarLink::HandleMessage(const FString& Message)
{
	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		return;
	}

	const FString Type = Root->GetStringField(TEXT("type"));
	if (Type == TEXT("motion"))
	{
		HandleMotion(Root);
	}
	else if (Type == TEXT("llm_response"))
	{
		// 동작은 서버가 같은 답으로 만든 motion 메시지로 따로 온다. 여기서는 알림만.
		TArray<FString> Gloss;
		Root->TryGetStringArrayField(TEXT("gloss"), Gloss);
		OnTutorReply.Broadcast(Root->GetStringField(TEXT("text")), Gloss);
	}
	else if (Type == TEXT("recognition"))
	{
		double Confidence = 0.0;
		Root->TryGetNumberField(TEXT("confidence"), Confidence);
		OnRecognition.Broadcast(Root->GetStringField(TEXT("text")), static_cast<float>(Confidence));
	}
}

void UHandTalkAvatarLink::HandleMotion(const TSharedPtr<FJsonObject>& Root)
{
	const TArray<TSharedPtr<FJsonValue>>* BoneItems = nullptr;
	const TArray<TSharedPtr<FJsonValue>>* FrameItems = nullptr;
	if (!LivePlayer.IsValid() || !Root->TryGetArrayField(TEXT("bones"), BoneItems) || !Root->TryGetArrayField(TEXT("frames"), FrameItems))
	{
		return;
	}
	TSharedRef<FHandTalkLiveMotion> Motion = MakeShared<FHandTalkLiveMotion>();
	for (const TSharedPtr<FJsonValue>& B : *BoneItems) Motion->Bones.Add(FName(*B->AsString()));
	const int32 NB = Motion->Bones.Num();
	Motion->Rotations.Reserve(FrameItems->Num() * NB);
	for (const TSharedPtr<FJsonValue>& F : *FrameItems)
	{
		const TArray<TSharedPtr<FJsonValue>>& V = F->AsArray();
		if (V.Num() != NB * 4) continue;
		for (int32 i = 0; i < NB; ++i)
		{
			Motion->Rotations.Add(FQuat(V[i * 4]->AsNumber(), V[i * 4 + 1]->AsNumber(), V[i * 4 + 2]->AsNumber(), V[i * 4 + 3]->AsNumber()));
		}
		++Motion->NumFrames;
	}
	if (Motion->NumFrames == 0)
	{
		return;
	}
	double Fps = 30.0, Gap = 0.0;
	Root->TryGetNumberField(TEXT("fps"), Fps);
	Root->TryGetNumberField(TEXT("loop_gap"), Gap);
	Root->TryGetBoolField(TEXT("loop"), Motion->bLoop);
	Motion->Fps = static_cast<float>(Fps);
	Motion->LoopGap = static_cast<float>(Gap);

	if (UWorld* World = GetWorld())
	{
		LivePlayer->Play(Motion, World->GetTimeSeconds());
	}
	UE_LOG(LogHandTalk, Log, TEXT("motion %d frames %.2fs%s"), Motion->NumFrames, Motion->Duration(), Motion->bLoop ? TEXT(" loop") : TEXT(""));
	if (!Motion->bLoop)
	{
		OnMotionStarted.Broadcast(Motion->Duration());
	}
}

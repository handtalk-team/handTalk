// handTalk — 서버 /ws/avatar 에 붙어 튜터 응답(avatar_commands)과 학습 단계(lesson)를 받아 수어 클립을 실시간 재생한다.

#include "HandTalkAvatarLink.h"

#include "Animation/AnimSequenceBase.h"
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
	Rest();
}

void UHandTalkAvatarLink::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Disconnect();
	Super::EndPlay(EndPlayReason);
}

void UHandTalkAvatarLink::SetupParts()
{
	Parts.Reset();
	AActor* Target = TargetActor ? TargetActor.Get() : GetOwner();
	if (!Target)
	{
		return;
	}

	TArray<USkeletalMeshComponent*> Meshes;
	Target->GetComponents(Meshes);
	for (USkeletalMeshComponent* Mesh : Meshes)
	{
		const FName Name = Mesh->GetFName();
		const FString* Folder = PartClipFolders.Find(Name);
		if (!Folder && PartClipFolders.Num() == 0 && Name == TEXT("Body"))
		{
			Folder = &ClipFolder;
		}
		if (!Folder)
		{
			continue;
		}
		// 리타깃/후처리 AnimBP 대신 클립을 그대로 재생 (옷 메시는 각자 클립을 받으므로 Leader Pose 해제)
		Mesh->SetLeaderPoseComponent(nullptr);
		Mesh->SetAnimationMode(EAnimationMode::AnimationSingleNode);
		if (Name == TEXT("Body"))
		{
			Mesh->SetDisablePostProcessBlueprint(true);
		}
		Parts.Add({ Mesh, *Folder });
	}
	UE_LOG(LogHandTalk, Log, TEXT("avatar parts: %d on %s"), Parts.Num(), *Target->GetName());
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
		World->GetTimerManager().ClearTimer(ClipTimer);
		World->GetTimerManager().ClearTimer(DemoTimer);
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
	if (Type == TEXT("llm_response"))
	{
		TArray<FHandTalkAvatarCommand> Commands;
		const TArray<TSharedPtr<FJsonValue>>* Items = nullptr;
		if (Root->TryGetArrayField(TEXT("avatar_commands"), Items))
		{
			for (const TSharedPtr<FJsonValue>& Item : *Items)
			{
				const TSharedPtr<FJsonObject> Obj = Item->AsObject();
				if (!Obj.IsValid()) continue;
				FHandTalkAvatarCommand Cmd;
				Cmd.Clip = FName(*Obj->GetStringField(TEXT("clip")));
				Obj->TryGetNumberField(TEXT("blend_in"), Cmd.BlendIn);
				Obj->TryGetNumberField(TEXT("blend_out"), Cmd.BlendOut);
				Obj->TryGetNumberField(TEXT("speed"), Cmd.Speed);
				Obj->TryGetStringField(TEXT("expression"), Cmd.Expression);
				Commands.Add(Cmd);
			}
		}
		OnTutorReply.Broadcast(Root->GetStringField(TEXT("text")), Commands);
		PlayCommands(Commands);
	}
	else if (Type == TEXT("lesson"))
	{
		// 연습 중인 목표 동작 → 쉬는 동안 반복 (프리토킹·완료 시에는 비움)
		TArray<FName> Clips;
		bool bDone = false;
		Root->TryGetBoolField(TEXT("done"), bDone);
		const TSharedPtr<FJsonObject>* Target = nullptr;
		if (!bDone && Root->TryGetObjectField(TEXT("target"), Target) && Target)
		{
			const TArray<TSharedPtr<FJsonValue>>* Items = nullptr;
			if ((*Target)->TryGetArrayField(TEXT("clips"), Items))
			{
				for (const TSharedPtr<FJsonValue>& V : *Items) Clips.Add(FName(*V->AsString()));
			}
		}
		SetDemoClips(Clips);
	}
	else if (Type == TEXT("recognition"))
	{
		double Confidence = 0.0;
		Root->TryGetNumberField(TEXT("confidence"), Confidence);
		OnRecognition.Broadcast(Root->GetStringField(TEXT("text")), static_cast<float>(Confidence));
	}
}

void UHandTalkAvatarLink::PlayCommands(const TArray<FHandTalkAvatarCommand>& Commands)
{
	// 새 응답이 오면 이전 응답의 남은 클립은 버린다.
	Queue = Commands;
	if (UWorld* World = GetWorld())
	{
		World->GetTimerManager().ClearTimer(ClipTimer);
		World->GetTimerManager().ClearTimer(DemoTimer);
	}
	PlayNext();
}

void UHandTalkAvatarLink::PlayClipByName(FName Clip)
{
	FHandTalkAvatarCommand Cmd;
	Cmd.Clip = Clip;
	PlayCommands({ Cmd });
}

void UHandTalkAvatarLink::SetDemoClips(const TArray<FName>& Clips)
{
	DemoClips = Clips;
	UWorld* World = GetWorld();
	if (World && !bPlaying)
	{
		World->GetTimerManager().SetTimer(DemoTimer, this, &UHandTalkAvatarLink::DemoTick, 0.6f, false);
	}
}

void UHandTalkAvatarLink::DemoTick()
{
	if (bPlaying || DemoClips.Num() == 0 || DemoInterval <= 0.f)
	{
		return;
	}
	Queue.Reset();
	for (const FName& C : DemoClips)
	{
		FHandTalkAvatarCommand Cmd;
		Cmd.Clip = C;
		Queue.Add(Cmd);
	}
	PlayNext();
}

UAnimSequenceBase* UHandTalkAvatarLink::ResolveClip(const FString& Folder, FName Clip) const
{
	if (Clip.IsNone())
	{
		return nullptr;
	}
	const FString Name = Clip.ToString();
	const FString Path = FString::Printf(TEXT("%s/%s.%s"), *Folder, *Name, *Name);
	return LoadObject<UAnimSequenceBase>(nullptr, *Path, nullptr, LOAD_NoWarn | LOAD_Quiet);
}

bool UHandTalkAvatarLink::PlayOnParts(FName Clip, float Speed, float& OutLength)
{
	OutLength = 0.f;
	bool bAny = false;
	for (const FPart& Part : Parts)
	{
		USkeletalMeshComponent* Mesh = Part.Mesh.Get();
		UAnimSequenceBase* Anim = Mesh ? ResolveClip(Part.Folder, Clip) : nullptr;
		if (!Anim) continue;
		Mesh->PlayAnimation(Anim, false);
		Mesh->SetPlayRate(Speed);
		OutLength = FMath::Max(OutLength, Anim->GetPlayLength());
		bAny = true;
	}
	return bAny;
}

void UHandTalkAvatarLink::PlayNext()
{
	UWorld* World = GetWorld();
	if (!World)
	{
		return;
	}

	while (Queue.Num() > 0)
	{
		const FHandTalkAvatarCommand Cmd = Queue[0];
		Queue.RemoveAt(0);

		const float Speed = Cmd.Speed > KINDA_SMALL_NUMBER ? Cmd.Speed : 1.0f;
		float Length = 0.f;
		if (!PlayOnParts(Cmd.Clip, Speed, Length))
		{
			continue;   // 이 클립은 아직 없음
		}
		bPlaying = true;
		UE_LOG(LogHandTalk, Log, TEXT("play %s (%.2fs)"), *Cmd.Clip.ToString(), Length / Speed);
		OnClipStarted.Broadcast(Cmd.Clip);
		World->GetTimerManager().SetTimer(ClipTimer, this, &UHandTalkAvatarLink::PlayNext, FMath::Max(Length / Speed, 0.05f), false);
		return;
	}

	bPlaying = false;
	Rest();
	if (DemoClips.Num() > 0 && DemoInterval > 0.f)
	{
		World->GetTimerManager().SetTimer(DemoTimer, this, &UHandTalkAvatarLink::DemoTick, DemoInterval, false);
	}
}

void UHandTalkAvatarLink::Rest()
{
	if (RestClip.IsNone())
	{
		return;
	}
	for (const FPart& Part : Parts)
	{
		USkeletalMeshComponent* Mesh = Part.Mesh.Get();
		UAnimSequenceBase* Anim = Mesh ? ResolveClip(Part.Folder, RestClip) : nullptr;
		if (!Anim) continue;
		Mesh->PlayAnimation(Anim, false);
		Mesh->SetPosition(Anim->GetPlayLength(), false);
		Mesh->SetPlayRate(0.f);
	}
}

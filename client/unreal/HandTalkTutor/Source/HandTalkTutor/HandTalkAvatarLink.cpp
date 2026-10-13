// handTalk — 서버 /ws/avatar 에 붙어 튜터 응답(avatar_commands)을 받아 수어 클립을 재생한다.

#include "HandTalkAvatarLink.h"

#include "Animation/AnimSequenceBase.h"
#include "Components/SkeletalMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "IWebSocket.h"
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

	if (!TargetMesh && GetOwner())
	{
		TArray<USkeletalMeshComponent*> Meshes;
		GetOwner()->GetComponents(Meshes);
		for (USkeletalMeshComponent* Mesh : Meshes)
		{
			if (Mesh->GetName().Contains(TEXT("Body")))
			{
				TargetMesh = Mesh;
				break;
			}
		}
		if (!TargetMesh && Meshes.Num() > 0)
		{
			TargetMesh = Meshes[0];
		}
	}

	Connect();
	PlayIdle();
}

void UHandTalkAvatarLink::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Disconnect();
	Super::EndPlay(EndPlayReason);
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
		UE_LOG(LogHandTalk, Log, TEXT("closed %d %s"), Code, *Reason);
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
	}
	PlayNext();
}

void UHandTalkAvatarLink::PlayClipByName(FName Clip)
{
	FHandTalkAvatarCommand Cmd;
	Cmd.Clip = Clip;
	PlayCommands({ Cmd });
}

UAnimSequenceBase* UHandTalkAvatarLink::ResolveClip(FName Clip) const
{
	if (Clip.IsNone())
	{
		return nullptr;
	}
	const FString Name = Clip.ToString();
	const FString Path = FString::Printf(TEXT("%s/%s.%s"), *ClipFolder, *Name, *Name);
	return LoadObject<UAnimSequenceBase>(nullptr, *Path, nullptr, LOAD_NoWarn | LOAD_Quiet);
}

void UHandTalkAvatarLink::PlayNext()
{
	UWorld* World = GetWorld();
	if (!World || !TargetMesh)
	{
		return;
	}

	while (Queue.Num() > 0)
	{
		const FHandTalkAvatarCommand Cmd = Queue[0];
		Queue.RemoveAt(0);

		UAnimSequenceBase* Anim = ResolveClip(Cmd.Clip);
		if (!Anim)
		{
			UE_LOG(LogHandTalk, Warning, TEXT("clip not found: %s/%s"), *ClipFolder, *Cmd.Clip.ToString());
			continue;
		}

		const float Speed = Cmd.Speed > KINDA_SMALL_NUMBER ? Cmd.Speed : 1.0f;
		TargetMesh->PlayAnimation(Anim, false);
		TargetMesh->SetPlayRate(Speed);
		OnClipStarted.Broadcast(Cmd.Clip);

		const float Duration = Anim->GetPlayLength() / Speed;
		World->GetTimerManager().SetTimer(ClipTimer, this, &UHandTalkAvatarLink::PlayNext, FMath::Max(Duration, 0.05f), false);
		return;
	}

	PlayIdle();
}

void UHandTalkAvatarLink::PlayIdle()
{
	if (TargetMesh && IdleAnim)
	{
		TargetMesh->PlayAnimation(IdleAnim, true);
		TargetMesh->SetPlayRate(1.0f);
	}
}

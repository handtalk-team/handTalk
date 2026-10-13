// handTalk — 옷(상의, 하의, 신발) 메시가 메타휴먼 Body 를 따라 움직이게 Leader Pose 를 걸어 주는 도우미 액터

#include "HandTalkLeaderPose.h"

#include "Components/SkeletalMeshComponent.h"
#include "TimerManager.h"

AHandTalkLeaderPose::AHandTalkLeaderPose()
{
	PrimaryActorTick.bCanEverTick = false;
}

void AHandTalkLeaderPose::BeginPlay()
{
	Super::BeginPlay();
	Apply();
	// 메타휴먼 BP 의 BeginPlay 가 나중에 Leader Pose 를 다시 바꿀 수 있어 잠시 뒤 한 번 더 건다
	FTimerHandle Handle;
	GetWorldTimerManager().SetTimer(Handle, this, &AHandTalkLeaderPose::Apply, 0.1f, false);
}

void AHandTalkLeaderPose::Apply()
{
	if (!Target)
	{
		return;
	}

	TArray<USkeletalMeshComponent*> Meshes;
	Target->GetComponents(Meshes);

	USkeletalMeshComponent* Leader = nullptr;
	for (USkeletalMeshComponent* Mesh : Meshes)
	{
		if (Mesh->GetFName() == LeaderComponentName)
		{
			Leader = Mesh;
			break;
		}
	}
	if (!Leader)
	{
		return;
	}

	// 리타깃용 AnimBP 에는 시퀀서 슬롯이 없어 수어 클립이 무시되므로 단일 노드로 바꾼다
	if (bUseSingleNodeOnLeader)
	{
		Leader->SetAnimationMode(EAnimationMode::AnimationSingleNode);
	}
	// 예전 메타휴먼 Body 의 후처리 AnimBP(물리 노드)가 빠른 팔 동작에서 가슴/어깨 피부를 늘어뜨리므로 끈다
	if (bDisablePostProcessOnLeader)
	{
		Leader->SetDisablePostProcessBlueprint(true);
	}

	for (USkeletalMeshComponent* Mesh : Meshes)
	{
		if (Mesh != Leader && !Exclude.Contains(Mesh->GetFName()))
		{
			if (!bUseLeaderPose)
			{
				Mesh->SetLeaderPoseComponent(nullptr);
				Mesh->SetAnimationMode(EAnimationMode::AnimationSingleNode);
				continue;
			}
			Mesh->SetLeaderPoseComponent(Leader, true);
			// 천 시뮬레이션은 첫 프레임의 A포즈에서 시작하므로 리셋해 현재 포즈에 맞춘다
			Mesh->RecreateClothingActors();
			UE_LOG(LogTemp, Log, TEXT("[handtalk] leader pose %s -> %s (leader=%s)"), *Mesh->GetName(), *Leader->GetName(),
				Mesh->LeaderPoseComponent.IsValid() ? TEXT("ok") : TEXT("none"));
		}
	}
}

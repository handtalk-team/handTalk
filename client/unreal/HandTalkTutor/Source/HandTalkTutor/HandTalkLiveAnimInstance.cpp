#include "HandTalkLiveAnimInstance.h"

#include "Animation/AnimNodeBase.h"
#include "Engine/World.h"

void FHandTalkLiveMotion::Sample(double Elapsed, TArray<FQuat>& Out) const
{
	const int32 NB = Bones.Num();
	Out.SetNum(NB);
	if (NumFrames == 0 || NB == 0)
	{
		return;
	}
	double T = Elapsed;
	if (bLoop && NumFrames > 1)
	{
		T = FMath::Fmod(T, static_cast<double>(Duration() + LoopGap));
	}
	const double F = FMath::Clamp(T * Fps, 0.0, static_cast<double>(NumFrames - 1));
	const int32 A = FMath::FloorToInt32(F);
	const int32 B = FMath::Min(A + 1, NumFrames - 1);
	const float Alpha = static_cast<float>(F - A);
	for (int32 i = 0; i < NB; ++i)
	{
		Out[i] = FQuat::Slerp(Rotations[A * NB + i], Rotations[B * NB + i], Alpha);
	}
}

void FHandTalkLivePlayer::Switch(TSharedPtr<const FHandTalkLiveMotion> Motion, double Now)
{
	Previous = Current;
	PreviousStart = StartTime;
	Current = Motion;
	StartTime = Now;
}

void FHandTalkLivePlayer::Play(TSharedPtr<const FHandTalkLiveMotion> Motion, double Now)
{
	if (!Motion.IsValid())
	{
		return;
	}
	const bool bBusy = Current.IsValid() && !Current->bLoop && !Current->IsFinished(Now - StartTime);
	if (Motion->bLoop && bBusy)
	{
		Pending = Motion;
		return;
	}
	if (!Motion->bLoop)
	{
		Pending.Reset();
	}
	Switch(Motion, Now);
}

bool FHandTalkLivePlayer::Sample(double Now, TArray<FName>& OutBones, TArray<FQuat>& OutRot)
{
	if (Pending.IsValid() && (!Current.IsValid() || Current->IsFinished(Now - StartTime)))
	{
		Switch(Pending, Now);
		Pending.Reset();
	}
	if (!Current.IsValid())
	{
		return false;
	}
	OutBones = Current->Bones;
	Current->Sample(Now - StartTime, OutRot);

	// 동작이 바뀐 직후에는 이전 동작에서 부드럽게 넘어간다
	const float Blend = BlendTime > 0.f ? static_cast<float>((Now - StartTime) / BlendTime) : 1.f;
	if (Blend < 1.f && Previous.IsValid() && Previous->Bones == Current->Bones)
	{
		TArray<FQuat> From;
		Previous->Sample(Now - PreviousStart, From);
		const float W = FMath::SmoothStep(0.f, 1.f, Blend);
		for (int32 i = 0; i < OutRot.Num(); ++i)
		{
			OutRot[i] = FQuat::Slerp(From[i], OutRot[i], W);
		}
	}
	else
	{
		Previous.Reset();
	}
	return true;
}

FAnimInstanceProxy* UHandTalkLiveAnimInstance::CreateAnimInstanceProxy()
{
	return new FHandTalkLiveProxy(this);
}

void FHandTalkLiveProxy::PreUpdate(UAnimInstance* InAnimInstance, float DeltaSeconds)
{
	FAnimInstanceProxy::PreUpdate(InAnimInstance, DeltaSeconds);

	const UHandTalkLiveAnimInstance* Inst = Cast<UHandTalkLiveAnimInstance>(InAnimInstance);
	const UWorld* World = InAnimInstance ? InAnimInstance->GetWorld() : nullptr;
	if (!Inst || !Inst->Player.IsValid() || !World)
	{
		return;
	}
	TArray<FName> NewBones;
	if (!Inst->Player->Sample(World->GetTimeSeconds(), NewBones, Pose))
	{
		Pose.Reset();
		return;
	}
	if (NewBones != Bones)
	{
		Bones = MoveTemp(NewBones);
		Index.Reset();
		for (int32 i = 0; i < Bones.Num(); ++i)
		{
			Index.Add(Bones[i], i);
		}
	}
}

bool FHandTalkLiveProxy::Evaluate(FPoseContext& Output)
{
	// Output 은 이 메시의 참조 포즈로 초기화되어 있다. 구동 본만 컴포넌트 공간 회전을 맞추고 나머지는 참조 로컬 그대로.
	FCompactPose& P = Output.Pose;
	if (Pose.Num() == 0)
	{
		return true;
	}
	const FBoneContainer& BC = P.GetBoneContainer();
	const FReferenceSkeleton& RefSkel = BC.GetReferenceSkeleton();
	const int32 N = P.GetNumBones();
	TArray<FQuat> Comp;
	Comp.SetNumUninitialized(N);
	for (int32 i = 0; i < N; ++i)
	{
		const FCompactPoseBoneIndex I(i);
		const FCompactPoseBoneIndex Parent = BC.GetParentBoneIndex(I);
		const FQuat ParentRot = Parent.IsValid() ? Comp[Parent.GetInt()] : FQuat::Identity;
		const FName Name = RefSkel.GetBoneName(BC.MakeMeshPoseIndex(I).GetInt());
		if (const int32* Idx = Index.Find(Name))
		{
			const FQuat Target = Pose[*Idx];
			Comp[i] = Target;
			P[I].SetRotation((ParentRot.Inverse() * Target).GetNormalized());
		}
		else
		{
			Comp[i] = ParentRot * P[I].GetRotation();
		}
	}
	return true;
}

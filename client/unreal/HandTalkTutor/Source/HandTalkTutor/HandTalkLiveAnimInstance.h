// handTalk — 서버가 실시간으로 만든 동작(본 회전)을 뼈대에 바로 입히는 애님 인스턴스. 애님 그래프도 클립도 쓰지 않는다.

#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "Animation/AnimInstanceProxy.h"
#include "HandTalkLiveAnimInstance.generated.h"

/** 서버 {"type":"motion"} 한 편: 프레임마다 구동 본의 컴포넌트 공간 회전 (Body 메시 기준) */
struct FHandTalkLiveMotion
{
	TArray<FName> Bones;
	TArray<FQuat> Rotations;   // [Frame * Bones.Num() + Bone]
	int32 NumFrames = 0;
	float Fps = 30.f;
	bool bLoop = false;
	float LoopGap = 0.f;       // 반복 사이에 마지막 프레임(쉬는 자세)을 잡고 있는 시간

	float Duration() const { return NumFrames > 1 ? (NumFrames - 1) / Fps : 0.f; }
	bool IsFinished(double Elapsed) const { return !bLoop && Elapsed >= Duration(); }
	void Sample(double Elapsed, TArray<FQuat>& Out) const;
};

/** 지금 재생 중인 동작과 전환(블렌드)을 관리. 링크 컴포넌트가 갖고, 부위별 애님 인스턴스가 함께 읽는다. */
struct FHandTalkLivePlayer
{
	TSharedPtr<const FHandTalkLiveMotion> Current;
	TSharedPtr<const FHandTalkLiveMotion> Previous;   // 전환 중 섞을 이전 동작
	TSharedPtr<const FHandTalkLiveMotion> Pending;    // 답을 다 한 뒤 이어서 틀 반복 동작 (시범, 쉬는 자세)
	double StartTime = 0.0;
	double PreviousStart = 0.0;
	float BlendTime = 0.25f;

	/** 한 번 재생할 동작은 바로 틀고, 반복 동작은 지금 동작이 끝난 뒤에 튼다. */
	void Play(TSharedPtr<const FHandTalkLiveMotion> Motion, double Now);
	bool Sample(double Now, TArray<FName>& OutBones, TArray<FQuat>& OutRot);

private:
	void Switch(TSharedPtr<const FHandTalkLiveMotion> Motion, double Now);
};

UCLASS(Transient, NotBlueprintable)
class HANDTALKTUTOR_API UHandTalkLiveAnimInstance : public UAnimInstance
{
	GENERATED_BODY()

public:
	TSharedPtr<FHandTalkLivePlayer> Player;

protected:
	virtual FAnimInstanceProxy* CreateAnimInstanceProxy() override;
};

struct FHandTalkLiveProxy : public FAnimInstanceProxy
{
	explicit FHandTalkLiveProxy(UAnimInstance* InInstance) : FAnimInstanceProxy(InInstance) {}

	virtual void PreUpdate(UAnimInstance* InAnimInstance, float DeltaSeconds) override;
	virtual bool Evaluate(FPoseContext& Output) override;

private:
	TArray<FName> Bones;
	TArray<FQuat> Pose;
	TMap<FName, int32> Index;
};

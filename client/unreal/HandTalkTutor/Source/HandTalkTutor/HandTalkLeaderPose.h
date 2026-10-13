// handTalk — 옷(상의, 하의, 신발) 메시가 메타휴먼 Body 를 따라 움직이게 Leader Pose 를 걸어 주는 도우미 액터

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HandTalkLeaderPose.generated.h"

UCLASS()
class HANDTALKTUTOR_API AHandTalkLeaderPose : public AActor
{
	GENERATED_BODY()

public:
	AHandTalkLeaderPose();

	/** Leader Pose 를 걸 메타휴먼 액터 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TObjectPtr<AActor> Target;

	/** 따라갈 기준 메시 이름 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	FName LeaderComponentName = TEXT("Body");

	/** Face 는 자체 AnimBP 가 있으므로 기본으로 제외 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	TArray<FName> Exclude = { TEXT("Face") };

	/** Body 의 AnimBP 대신 시퀀서/PlayAnimation 클립을 그대로 재생 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	bool bUseSingleNodeOnLeader = true;

	/** Body 메시의 후처리 AnimBP 를 끈다 (예전 메타휴먼의 물리 노드 문제 회피) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	bool bDisablePostProcessOnLeader = false;

	/** false 면 Leader Pose 대신 옷 메시도 단일 노드로 두어 부위별 클립을 직접 재생한다 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HandTalk")
	bool bUseLeaderPose = true;

	UFUNCTION(BlueprintCallable, Category = "HandTalk")
	void Apply();

protected:
	virtual void BeginPlay() override;
};

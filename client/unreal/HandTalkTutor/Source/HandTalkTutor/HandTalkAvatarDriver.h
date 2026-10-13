// handTalk — 레벨에 놓는 실시간 튜터 구동기. 메타휴먼 BP 를 고치지 않고 TargetActor 로 지정해 움직인다.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HandTalkAvatarDriver.generated.h"

class UHandTalkAvatarLink;

UCLASS()
class HANDTALKTUTOR_API AHandTalkAvatarDriver : public AActor
{
	GENERATED_BODY()

public:
	AHandTalkAvatarDriver();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HandTalk")
	TObjectPtr<UHandTalkAvatarLink> Link;
};

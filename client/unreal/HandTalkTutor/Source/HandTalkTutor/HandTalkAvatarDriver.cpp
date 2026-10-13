// handTalk — 레벨에 놓는 실시간 튜터 구동기. 메타휴먼 BP 를 고치지 않고 TargetActor 로 지정해 움직인다.

#include "HandTalkAvatarDriver.h"

#include "HandTalkAvatarLink.h"

AHandTalkAvatarDriver::AHandTalkAvatarDriver()
{
	PrimaryActorTick.bCanEverTick = false;
	Link = CreateDefaultSubobject<UHandTalkAvatarLink>(TEXT("AvatarLink"));
}

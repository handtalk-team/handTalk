// handTalk — 에디터 파이썬(build_sign_anim.py)에서 스켈레톤 계층을 읽기 위한 함수

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "HandTalkSkeletonLibrary.generated.h"

class USkeleton;

UCLASS()
class HANDTALKTUTOR_API UHandTalkSkeletonLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** 참조 스켈레톤 순서대로 본 이름과 부모 이름(루트는 None)을 돌려준다. */
	UFUNCTION(BlueprintCallable, Category = "HandTalk")
	static void GetBoneHierarchy(const USkeleton* Skeleton, TArray<FName>& BoneNames, TArray<FName>& ParentNames);
};

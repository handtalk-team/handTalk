// handTalk — 에디터 파이썬(build_sign_anim.py)에서 스켈레톤 계층을 읽기 위한 함수

#include "HandTalkSkeletonLibrary.h"

#include "Animation/Skeleton.h"

void UHandTalkSkeletonLibrary::GetBoneHierarchy(const USkeleton* Skeleton, TArray<FName>& BoneNames, TArray<FName>& ParentNames)
{
	BoneNames.Reset();
	ParentNames.Reset();
	if (!Skeleton)
	{
		return;
	}

	const FReferenceSkeleton& RefSkeleton = Skeleton->GetReferenceSkeleton();
	const int32 Num = RefSkeleton.GetNum();
	for (int32 Index = 0; Index < Num; ++Index)
	{
		BoneNames.Add(RefSkeleton.GetBoneName(Index));
		const int32 Parent = RefSkeleton.GetParentIndex(Index);
		ParentNames.Add(Parent == INDEX_NONE ? NAME_None : RefSkeleton.GetBoneName(Parent));
	}
}

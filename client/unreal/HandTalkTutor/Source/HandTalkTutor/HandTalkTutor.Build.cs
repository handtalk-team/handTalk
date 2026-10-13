// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class HandTalkTutor : ModuleRules
{
	public HandTalkTutor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "EnhancedInput" });

		// 서버(FastAPI /ws/avatar) 연결과 JSON 메시지 파싱
		PrivateDependencyModuleNames.AddRange(new string[] { "WebSockets", "Json", "JsonUtilities" });
	}
}

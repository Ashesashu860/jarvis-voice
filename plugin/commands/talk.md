---
description: Hands-free voice conversation in this session (speak, hear the reply, speak again; ends after silence or "that's all")
allowed-tools: Bash(jarvis converse:*)
---
Voice conversation mode is now on for this session. After each reply, Jarvis reads it aloud and listens for the user's spoken follow-up, which will arrive as a stop-hook message; treat each one exactly as the user's next prompt. Keep the opening of every reply short and speakable.

The user spoke the following request aloud. It is a speech-to-text transcript, so it may contain recognition errors: infer the intended meaning. Treat it exactly as if the user had typed it as their prompt. If it reads "(no speech detected)", say in one short line that you didn't catch anything and that they can run /jarvis:talk again.

!`jarvis converse`

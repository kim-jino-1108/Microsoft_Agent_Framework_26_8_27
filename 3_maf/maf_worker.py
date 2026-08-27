"""Microsoft Agent Framework 워커를 서브프로세스로 실행해 보드(board) 작업을 수행한다.

인자 없이 실행하면 자체 Day 3 목표를 시드하고 처리한다(notes.txt 읽기 → 스페인어 번역
→ spanish.txt 쓰기): 보드에 단계를 계획하고, 파일시스템 MCP 서버로 파일을 읽고, 번역하고,
스페인어 결과를 파일로 쓰고, 각 단계를 완료 처리한 뒤 목표를 닫는다.

task id와 공유 보드 경로를 주면 같은 워커가 Day 5 에이전트 루프에 합류한다: 보드 도구와
파일 도구를 공유 보드/공유 폴더로 향하게 하고, 해당 태스크 하나만 클레임해서 요구 사항을
만들고 종료하며, 보드의 나머지 항목은 건드리지 않는다.

    uv run maf_worker.py                       # 단독 실행 Day 3 데모
    uv run maf_worker.py <taskId> <boardPath>  # Day 5: 공유 보드에서 태스크 하나만 처리
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from pathlib import Path

# Day 5 모드의 실행 형식은 "<taskId> <boardPath>"이다.
# board 모듈은 import 시점에 BOARD_PATH 환경변수를 읽어 사용할 파일을 결정하므로,
# board를 import 하기 전에 인자를 먼저 읽어 환경변수를 설정한다.
# 인자 없이 실행하면 아래 두 줄은 동작하지 않는다.
TASK_ID = int(sys.argv[1]) if len(sys.argv) > 2 else None
if TASK_ID is not None:
    os.environ.setdefault("BOARD_PATH", sys.argv[2])

# MAF가 import 시점에 출력하는 experimental 관련 경고 메시지를 숨긴다.
warnings.filterwarnings("ignore", message=r".*experimental.*")

from dotenv import load_dotenv  # noqa: E402
from agent_framework import Agent, MCPStdioTool  # noqa: E402
from agent_framework.openai import OpenAIChatClient  # noqa: E402

import board  # noqa: E402

# .env 파일의 값을 읽어 기존 환경변수를 덮어쓴다.
load_dotenv(override=True)

# 사용할 모델명. WORKER_MODEL 환경변수가 없으면 기본값을 사용한다.
MODEL = os.environ.get("WORKER_MODEL", "gpt-5.4-mini")

# 이 스크립트 파일이 있는 디렉터리 아래의 workspace 폴더 경로
WORKSPACE = Path(__file__).resolve().parent / "workspace"

# 단독 실행 시 에이전트에게 부여되는 목표 문장
GOAL = "notes.txt를 읽고, 그 내용을 자연스러운 스페인어로 번역한 뒤, 그 스페인어를 spanish.txt에 작성하라."

# 파일 도구가 쓰기를 수행할 디렉터리.
# 단독 실행이면 이 워커의 workspace, Day 5 태스크 처리 중이면 보드 파일이 있는 공유 폴더.
WORK_DIR = WORKSPACE if TASK_ID is None else Path(sys.argv[2]).resolve().parent

# 지정한 모델로 OpenAI 채팅 클라이언트를 생성한다.
client = OpenAIChatClient(model=MODEL)


def show_todos() -> list[dict]:
    """보드에 있는 모든 todo 항목을 나열한다. 목표(goal)는 parent_id가 None이고, 단계(step)는 parent_id에 소속 목표의 id가 들어 있다."""
    # 보드 모듈에서 전체 todo 목록을 가져와 그대로 반환한다.
    return board.list_todos()


def plan_steps(goal_id: int, steps: list[str]) -> dict:
    """하나의 목표를 순서가 있는 단계 체크리스트로 나누어 보드에 등록한다. 목표의 id와 짧은 단계 설명 목록을 전달한다."""
    # 전달받은 각 단계 설명을 해당 목표의 하위 단계로 추가하고, 생성된 단계 id 목록을 반환한다.
    return {"goal_id": goal_id, "step_ids": [board.add_step(goal_id, step) for step in steps]}


def complete_task(task_id: int, result: str) -> dict:
    """해당 id의 todo(단계 또는 목표)를 완료 상태로 표시하고 짧은 결과 요약을 기록한다."""
    # 보드에서 해당 항목을 완료 처리한다.
    board.complete_todo(task_id, result)
    return {"task_id": task_id, "status": "done"}


# 에이전트에게 주어지는 시스템 지시문
INSTRUCTIONS = """
너는 공유 todo 보드와 여러 파일 도구를 다루는 신중한 작업자다.

대기 중인 목표를 맡아 끝까지 완수하라. 먼저 짧은 계획을 세우는 것부터 시작하라. 즉, 그 작업이 나뉘는 구체적인 몇 개의 단계를 보드의 해당 목표 아래에 추가하라. 그런 다음 파일 도구로 그 단계들을 수행하고, 각 단계를 끝낼 때마다 완료로 표시하라. 모든 단계가 끝나면 목표를 닫아라. 네 파일들은 도구가 사용할 수 있도록 허용된 단 하나의 폴더 안에 있다.
"""


def seed() -> int:
    """보드를 초기화하고, 이전 출력 파일을 지우고, 목표 하나를 추가한다."""
    # 보드를 비운다.
    board.reset_board()
    # 작업 폴더가 없으면 생성한다.
    WORKSPACE.mkdir(exist_ok=True)
    # 이전 실행에서 생성된 출력 파일이 있으면 삭제한다(없어도 오류를 내지 않는다).
    (WORKSPACE / "spanish.txt").unlink(missing_ok=True)
    # 목표를 보드에 추가하고 id를 받는다.
    goal_id = board.add_goal(GOAL)
    # 워커가 해당 목표를 집어 든다: pending -> in_progress 상태로 변경된다.
    board.claim_todo(goal_id)
    return goal_id


async def main() -> None:
    if TASK_ID is None:
        # 단독 실행 모드: 목표를 시드하고 그 목표를 처리하라고 지시한다.
        goal_id = seed()
        print(f"목표 {goal_id} 시드 완료: {GOAL}\n")
        message = "보드에 대기 중인 목표를 처리해 줘."
    else:
        # Day 5 모드: 공유 보드에서 이 태스크 하나만 in_progress로 전환한다.
        board.claim_todo(TASK_ID)
        message = (
            f"너는 공유 보드에서 태스크 #{TASK_ID}를 클레임했다. 오직 그 태스크와 그 하위 단계만 처리하라. "
            f"작업을 만들고 확인까지 마쳤으면, complete_task로 태스크 #{TASK_ID} 자체를 완료 처리한 뒤 멈춰라."
        )

    # 파일시스템 MCP 서버를 stdio 방식으로 실행하는 도구를 구성한다.
    filesystem = MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(WORK_DIR)],
        # 서버를 작업 디렉터리에서 시작해, 상대 경로 파일명이 그 폴더 기준으로 해석되게 한다.
        cwd=str(WORK_DIR),
    )
    # with 블록을 벗어나면 MCP 서버 프로세스가 정리된다.
    async with filesystem:
        # 지시문과 도구 목록을 지정해 에이전트를 생성한다.
        worker = Agent(
            client=client,
            instructions=INSTRUCTIONS,
            tools=[show_todos, plan_steps, complete_task, filesystem],
        )
        # 준비한 메시지를 입력으로 에이전트를 실행한다.
        await worker.run(message)

    # 단독 실행일 때만 결과를 출력한다. Day 5에서는 오케스트레이터가 콘솔 출력을 담당한다.
    if TASK_ID is None:
        print("\n실행 후 보드 상태:")
        board.show_board()
        spanish = WORKSPACE / "spanish.txt"
        # 출력 파일이 생성되었으면 내용을 화면에 표시한다.
        if spanish.exists():
            print("\nspanish.txt:\n" + spanish.read_text(encoding="utf-8"))


if __name__ == "__main__":
    # 비동기 main 코루틴을 실행한다.
    asyncio.run(main())
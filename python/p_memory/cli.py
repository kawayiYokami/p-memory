"""CLI for interactive testing, health checks and read-only legacy import."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
import sys

from .api import KnowledgeBase, import_legacy
from .agent import AgentError, OpenAICompatibleClient, SimpleAgent
from .errors import PMemoryError


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="p-memory", description="Memory core and a simple OpenAI-compatible agent")
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="Run an interactive or one-shot simple agent")
    chat.add_argument("--data", required=True, help="Independent data directory for this agent")
    chat.add_argument("--model", default=os.environ.get("P_MEMORY_MODEL") or os.environ.get("OPENAI_MODEL"))
    chat.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    chat.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key")
    chat.add_argument("--namespace", default="default")
    chat.add_argument("--scope", default="public", help="Write scope and default read scope")
    chat.add_argument("--read-scope", action="append", help="Explicit read scopes; may be repeated")
    chat.add_argument("--prompt", help="One-shot prompt; omit for interactive mode")
    chat.add_argument("--read-only", action="store_true", help="Expose only retrieval tools")
    chat.add_argument("--max-rounds", type=int, default=8)
    chat.add_argument("--timeout", type=float, default=60)
    chat.add_argument("--trace", action="store_true", help="Print tool traces to stderr")
    chat.add_argument("--json", action="store_true", help="Return a JSON result for one-shot mode")
    chat.add_argument("--embedding-model", help="Optional OpenAI-compatible embedding model")
    chat.add_argument("--embedding-dimension", type=int)
    chat.add_argument("--embedding-space", default="simple-agent-v1")
    for name in ("health", "rebuild"):
        p = commands.add_parser(name)
        p.add_argument("--data", required=True)
    migrate = commands.add_parser("import", help="Preview or apply a legacy snapshot to a new directory")
    migrate.add_argument("--source", required=True, choices=("p_ai", "world_tree", "angel_memory"))
    migrate.add_argument("--source-id", required=True)
    migrate.add_argument("--database", required=True)
    migrate.add_argument("--destination", required=True)
    migrate.add_argument("--graph-database")
    migrate.add_argument("--lookup-database")
    migrate.add_argument("--notes-root")
    migrate.add_argument("--namespace", default="default")
    migrate.add_argument("--scope", default="public")
    migrate.add_argument("--apply", action="store_true", help="Write validated data; default is a dry run")
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            result = import_legacy(source=args.source, source_id=args.source_id, database=args.database, destination=args.destination,
                                   graph_database=args.graph_database, lookup_database=args.lookup_database, notes_root=args.notes_root,
                                   namespace=args.namespace, scope=args.scope, dry_run=not args.apply)
            _print(result)
            return 2 if result["conflicts"] else 0
        if args.command in ("health", "rebuild"):
            with KnowledgeBase(args.data) as kb:
                _print(kb.health() if args.command == "health" else kb.rebuild_indexes())
            return 0
        if not args.model:
            parser.error("chat requires --model or P_MEMORY_MODEL / OPENAI_MODEL")
        client = OpenAICompatibleClient(model=args.model, base_url=args.base_url,
                                        api_key=os.environ.get(args.api_key_env, ""), timeout=args.timeout)
        with KnowledgeBase(args.data, namespace=args.namespace, scopes=args.read_scope or [args.scope], write_scope=args.scope) as kb:
            agent = SimpleAgent(kb, client, read_only=args.read_only, max_rounds=args.max_rounds,
                                embedding_model=args.embedding_model, embedding_dimension=args.embedding_dimension,
                                embedding_space=args.embedding_space)

            def answer(prompt: str) -> None:
                result = agent.run(prompt)
                if args.trace:
                    for trace in result.tool_traces:
                        print(json.dumps(asdict(trace), ensure_ascii=False), file=sys.stderr)
                if args.json:
                    _print(asdict(result))
                else:
                    print(result.content)

            if args.prompt is not None:
                answer(args.prompt)
            else:
                print("p-memory simple agent — /exit 退出，/reset 清空本轮对话，记忆仍保留。")
                while True:
                    try:
                        prompt = input("> ").strip()
                    except EOFError:
                        break
                    if prompt in ("/exit", "/quit"):
                        break
                    if prompt == "/reset":
                        agent.reset()
                    elif prompt:
                        try:
                            answer(prompt)
                        except (AgentError, PMemoryError) as exc:
                            print(f"Error: {exc}", file=sys.stderr)
        return 0
    except (PMemoryError, AgentError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

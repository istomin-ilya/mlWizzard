import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.orchestrator import run_agent


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Wizard CLI",
    )

    parser.add_argument(
        "query",
        type=str,
        help="Natural language query for the ML Wizard agent.",
    )

    args = parser.parse_args()

    console.print(
        Panel.fit(
            args.query,
            title="ML Wizard Query",
            border_style="blue",
        )
    )

    try:
        answer = run_agent(args.query)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)

    console.print(Markdown(answer))


if __name__ == "__main__":
    main()
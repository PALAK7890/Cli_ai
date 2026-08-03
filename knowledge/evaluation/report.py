"""
Pretty benchmark reporting.
"""

from rich.console import Console

console = Console()


def print_report(metrics: dict):

    console.print("\n[bold cyan]Benchmark Report[/bold cyan]\n")

    for key, value in metrics.items():

        if key == "Latency":
            console.print(
                f"{key:<15}: {value:.2f} ms"
            )

        else:
            console.print(
                f"{key:<15}: {value:.4f}"
            )
"""Interactive terminal client for the chat API.

    python -m scripts.chat_cli [--url http://127.0.0.1:8000] [-q QUESTION]

Streams `POST /chat/stream` and renders the SSE events as they arrive. When the
graph pauses on the human-in-the-loop gate, the pending action is shown here and
the decision goes to `POST /approvals/{thread_id}` — so a full approve/edit/
reject cycle happens without leaving the terminal.

`tenant_id` is never sent: it comes from the server's `Principal`. Under
`AUTH_MODE=jwt`, export `RAGBOT_TOKEN` and it goes out as a bearer token.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterator
from typing import Any

import httpx

with contextlib.suppress(ImportError):  # arrow keys / history in input()
    import readline  # noqa: F401

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def dim(t: str) -> str:
    return _c("2", t)


def bold(t: str) -> str:
    return _c("1", t)


def red(t: str) -> str:
    return _c("31", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


BANNER = f"""{bold("RAG support bot")} {dim("— /help for commands, /exit to quit")}"""

HELP = """
  /new           start a new thread (forget the conversation so far)
  /thread        show the current thread_id
  /nodes         toggle graph node names as they run
  /sources       re-print the citations of the last answer
  /help          this help
  /exit          quit
"""


class ChatError(RuntimeError):
    pass


class Client:
    def __init__(self, url: str, token: str | None, stream: bool) -> None:
        self.url = url.rstrip("/")
        self.stream = stream
        headers = {"content-type": "application/json"}
        if token:
            headers["authorization"] = f"Bearer {token}"
        # No read timeout: a local model can think for minutes.
        self.http = httpx.Client(
            headers=headers, timeout=httpx.Timeout(10.0, read=None, write=30.0)
        )
        self.thread_id: str | None = None
        self.citations: list[dict[str, Any]] = []
        self.show_nodes = False

    def close(self) -> None:
        self.http.close()

    # -- transport ---------------------------------------------------------
    def _payload(self, question: str) -> dict[str, Any]:
        body: dict[str, Any] = {"question": question}
        if self.thread_id:
            body["thread_id"] = self.thread_id
        return body

    def _sse(self, question: str) -> Iterator[tuple[str, dict[str, Any]]]:
        with self.http.stream(
            "POST", f"{self.url}/chat/stream", json=self._payload(question)
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise ChatError(_http_error(response))
            event, data = "message", []
            for line in response.iter_lines():
                line = line.rstrip("\r")
                if not line:
                    if data:
                        yield event, json.loads("\n".join(data) or "{}")
                    event, data = "message", []
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())

    # -- one turn ----------------------------------------------------------
    def ask(self, question: str) -> None:
        if self.stream:
            self._ask_streaming(question)
        else:
            self._ask_blocking(question)

    def _ask_streaming(self, question: str) -> None:
        printed_any = False
        for event, data in self._sse(question):
            if event == "token":
                sys.stdout.write(data.get("text", ""))
                sys.stdout.flush()
                printed_any = True
            elif event == "node" and self.show_nodes:
                print(dim(f"[{data.get('node')}]"))
            elif event == "answer":
                self.thread_id = data.get("thread_id") or self.thread_id
                answer = data.get("answer", "")
                if not printed_any and answer:
                    print(answer)
                elif printed_any:
                    print()
                if not answer and not printed_any:
                    print(yellow("(empty answer)"))
                self.citations = data.get("citations") or []
                self._print_sources()
                self._print_trace(data.get("trace_id"))
            elif event == "approval":
                self.thread_id = data.get("thread_id") or self.thread_id
                if printed_any:
                    print()
                self._handle_approval(data.get("pending_action") or {})
            elif event == "error":
                print(red(f"\nerror: {data.get('detail')}"))

    def _ask_blocking(self, question: str) -> None:
        response = self.http.post(f"{self.url}/chat", json=self._payload(question))
        if response.status_code >= 400:
            raise ChatError(_http_error(response))
        data = response.json()
        self.thread_id = data.get("thread_id") or self.thread_id
        if data.get("approval_required"):
            self._handle_approval(data.get("pending_action") or {})
            return
        print(data.get("answer") or yellow("(empty answer)"))
        self.citations = data.get("citations") or []
        self._print_sources()
        self._print_trace(data.get("trace_id"))

    # -- rendering ---------------------------------------------------------
    def _print_sources(self) -> None:
        if not self.citations:
            return
        print(dim("\nsources:"))
        for c in self.citations:
            print(
                dim(
                    f"  [{c.get('index')}] {c.get('citation')} "
                    f"({c.get('source_path')}, score {c.get('score', 0):.3f})"
                )
            )

    def _print_trace(self, trace_id: str | None) -> None:
        if trace_id:
            print(dim(f"trace: {trace_id}"))

    # -- human-in-the-loop -------------------------------------------------
    def _handle_approval(self, action: dict[str, Any]) -> None:
        tool = action.get("tool", "?")
        arguments = action.get("arguments") or {}
        reason = action.get("reason")

        print(yellow("\n[!] the agent is asking for approval"))
        print(f"  tool:      {bold(tool)}")
        print(f"  arguments: {json.dumps(arguments, ensure_ascii=False)}")
        if reason:
            print(f"  reason:    {reason}")

        while True:
            choice = _prompt(cyan("  [a]pprove / [r]eject / [e]dit / [s]kip > ")).strip().lower()
            if choice in {"a", "approve", "y", "yes"}:
                self._decide("approve", None)
                return
            if choice in {"r", "reject", "n", "no"}:
                note = _prompt("  reason for rejecting (optional): ").strip() or None
                self._decide("reject", None, note=note)
                return
            if choice in {"e", "edit"}:
                raw = _prompt(
                    f"  new arguments as JSON [{json.dumps(arguments, ensure_ascii=False)}]: "
                ).strip()
                if not raw:
                    print(dim("  unchanged"))
                    continue
                try:
                    edited = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(red(f"  invalid JSON: {exc}"))
                    continue
                if not isinstance(edited, dict):
                    print(red("  expected a JSON object"))
                    continue
                self._decide("edit", edited)
                return
            if choice in {"s", "skip"}:
                print(dim(f"  left pending — decide later: GET {self.url}/approvals"))
                return

    def _decide(
        self, decision: str, arguments: dict[str, Any] | None, note: str | None = None
    ) -> None:
        body: dict[str, Any] = {"decision": decision}
        if arguments is not None:
            body["arguments"] = arguments
        if note:
            body["note"] = note

        response = self.http.post(f"{self.url}/approvals/{self.thread_id}", json=body)
        if response.status_code >= 400:
            print(red(f"  could not send the decision: {_http_error(response)}"))
            return

        data = response.json()
        status = data.get("status")
        print(green(f"  -> {status}") if status == "approved" else yellow(f"  -> {status}"))
        for executed in data.get("executed") or []:
            print(dim(f"  executed: {json.dumps(executed, ensure_ascii=False)}"))
        if data.get("answer"):
            print()
            print(data["answer"])
        self._print_trace(data.get("trace_id"))


def _http_error(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if isinstance(detail, list):  # pydantic 422
        detail = "; ".join(
            f"{'.'.join(str(x) for x in d.get('loc', []))}: {d.get('msg')}" for d in detail
        )
    return f"HTTP {response.status_code}: {detail}"


def _prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return ""


def _handle_command(client: Client, line: str) -> bool:
    """Return True if the REPL should keep going, False to quit."""
    command = line.split()[0].lower()
    if command in {"/exit", "/quit", "/q"}:
        return False
    if command == "/help":
        print(HELP)
    elif command == "/new":
        client.thread_id = None
        client.citations = []
        print(dim("new thread"))
    elif command == "/thread":
        print(dim(client.thread_id or "no thread started yet"))
    elif command == "/nodes":
        client.show_nodes = not client.show_nodes
        print(dim(f"graph nodes: {'shown' if client.show_nodes else 'hidden'}"))
    elif command == "/sources":
        if client.citations:
            client._print_sources()
        else:
            print(dim("no citations yet"))
    else:
        print(dim(f"unknown command {command} — try /help"))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("RAGBOT_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("-q", "--question", help="ask once, print the answer, exit")
    parser.add_argument("--no-stream", action="store_true", help="use POST /chat instead of SSE")
    parser.add_argument("--nodes", action="store_true", help="show graph node names as they run")
    args = parser.parse_args(argv)

    client = Client(args.url, os.environ.get("RAGBOT_TOKEN"), stream=not args.no_stream)
    client.show_nodes = args.nodes

    try:
        if args.question:
            client.ask(args.question)
            return 0

        print(BANNER)
        while True:
            try:
                line = input(cyan("\n> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if not _handle_command(client, line):
                    return 0
                continue
            try:
                client.ask(line)
            except KeyboardInterrupt:
                print(red("\ninterrupted"))
            except ChatError as exc:
                print(red(f"\n{exc}"))
            except httpx.HTTPError as exc:
                print(red(f"\nnetwork: {exc}"))
    except ChatError as exc:
        print(red(str(exc)), file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(red(f"could not reach {args.url}: {exc}"), file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())

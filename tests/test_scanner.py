from __future__ import annotations

from token_eval.scanner import scan_repo

PY_APP = '''
"""Module docstring that is long enough to look like a prompt but must be ignored by the scanner because it is documentation, not a prompt sent to any model at all."""
import anthropic

SYSTEM_PROMPT = """You are a meticulous research assistant. Always cite sources.
Answer in at most three paragraphs and never speculate beyond the provided documents."""

client = anthropic.Anthropic()

def ask(q):
    return client.messages.create(model="claude-opus-5-5", max_tokens=1024,
                                  system=SYSTEM_PROMPT, messages=[{"role": "user", "content": q}])
'''

TS_APP = '''
import OpenAI from "openai";
const openai = new OpenAI();
const instructions = `You are a helpful shopping assistant for an online store. Recommend products
based on the user's stated budget and preferences, and always mention the return policy. Keep
answers under 100 words and use a friendly tone. Never invent product names or prices.`;
export async function chat(q: string) {
  return openai.chat.completions.create({ model: "gpt-4o-mini", messages: [
    { role: "system", content: instructions }, { role: "user", content: q }] });
}
'''


def make_repo(tmp_path):
    (tmp_path / "app.py").write_text(PY_APP)
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "chat.ts").write_text(TS_APP)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "summarize.txt").write_text("Summarize the following text: {text}")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text('x.messages.create({model: "claude-x"})')
    return tmp_path


def test_scan_finds_calls_models_and_prompts(tmp_path):
    r = scan_repo(make_repo(tmp_path))
    kinds = {(c.file.replace("\\", "/"), c.provider, c.kind) for c in r.call_sites}
    assert kinds == {("app.py", "anthropic", "messages.create"),
                     ("web/chat.ts", "openai", "chat.completions")}
    assert r.models == {"claude-opus-5-5": 1, "gpt-4o-mini": 1}
    locs = [p.location.replace("\\", "/") for p in r.prompts]
    assert any(l.startswith("app.py:") for l in locs)      # SYSTEM_PROMPT
    assert any(l.startswith("web/chat.ts:") for l in locs)  # template literal
    assert "prompts/summarize.txt" in locs
    assert not any("docstring" in p.preview for p in r.prompts)
    assert r.static_prompt_tokens == sum(p.tokens for p in r.prompts) > 0
    assert r.static_prompt_cost_usd > 0
    assert r.files_scanned == 3  # node_modules skipped
    assert r.codebase_tokens == sum(r.codebase_tokens_by_ext.values()) > 0


def test_scan_empty_repo_has_note(tmp_path):
    (tmp_path / "README.md").write_text("hello")
    r = scan_repo(tmp_path)
    assert r.call_sites == [] and any("No known LLM" in n for n in r.notes)


def test_js_lexer_does_not_join_code_between_templates():
    from token_eval.scanner import js_strings

    src = 'const a = `x ${y} z`; // `not a string`\nfoo(); /* `nor this` */ const b = "q\\"t";'
    assert [s for _, s in js_strings(src)] == ["x  z", 'q"t']


def test_noise_is_filtered(tmp_path):
    (tmp_path / "a.py").write_text(
        'def f():\n    """Calls client.messages.create() with params."""\n'
        '    # client.chat.completions.create(...)\n'
        '    cfg = "claude-lock.json"\n    name = "claude_managed_agents"\n'
        '    css = "flex items-center justify-between gap-2 px-4 py-2 rounded-md border '
        'text-sm font-medium hover:bg-accent " * 3\n')
    r = scan_repo(tmp_path)
    assert r.call_sites == [] and r.models == {} and r.prompts == []

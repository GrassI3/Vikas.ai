"""Test the full LangGraph pipeline end-to-end."""
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

async def main():
    from backend.agents.graph import run_pipeline

    state = {
        "session_id": "test-002",
        "user_utterance": "I have a severe headache",
        "translated_utterance": "I have a severe headache",
        "detected_language": "en",
        "conversation_history": [],
        "turn_count": 1,
    }

    print("\n[TEST] Running full LangGraph pipeline...")
    result = await run_pipeline(state)

    print("\n" + "="*60)
    print("FULL PIPELINE RESULT:")
    print(f"  severity        = {result.get('severity')}")
    print(f"  domain          = {result.get('domain')}")
    print(f"  confidence      = {result.get('confidence')}")
    print(f"  reasoning_chain = {result.get('reasoning_chain')}")
    print(f"  citations       = {result.get('citations')}")
    print(f"  disclaimer      = {result.get('disclaimer_injected')}")
    print(f"  response        = {result.get('final_response_en', '')[:300]}")
    print("="*60)

    # Validate
    ok = True
    if not result.get("reasoning_chain"):
        print("[FAIL] reasoning_chain is empty!")
        ok = False
    if not result.get("citations"):
        print("[FAIL] citations is empty!")
        ok = False
    if result.get("confidence", 0) == 0:
        print("[FAIL] confidence is 0!")
        ok = False
    if ok:
        print("\n[PASS] All checks passed! Pipeline is working correctly.")

if __name__ == "__main__":
    asyncio.run(main())

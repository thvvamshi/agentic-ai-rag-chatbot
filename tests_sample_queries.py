from src.graph import rag_graph
import time


TEST_QUERIES = [
    {
        "id": 1,
        "name": "Basic definition",
        "query": (
            "What is Agentic AI according to the document, and what "
            "makes an AI system agentic rather than simply an AI system "
            "that generates content?"
        ),
        "expected": "answer",
    },
    {
        "id": 2,
        "name": "Six core pillars",
        "query": (
            "What are the six core pillars of Agentic AI described in "
            "the document? Explain the purpose and role of each pillar."
        ),
        "expected": "answer",
    },
    {
        "id": 3,
        "name": "Perception to execution",
        "query": (
            "How does an Agentic AI system move from perceiving its "
            "environment to taking action? Explain the stages involved "
            "between perception and execution."
        ),
        "expected": "answer",
    },
    {
        "id": 4,
        "name": "LLM vs Agentic AI",
        "query": (
            "Why does the document say that agents are more than LLMs? "
            "Explain the difference between an LLM and an Agentic AI "
            "system in terms of goals, autonomy, decision-making, and action."
        ),
        "expected": "answer",
    },
    {
        "id": 5,
        "name": "Layered architecture",
        "query": (
            "Describe the layered architecture of an Agentic AI system "
            "from environmental perception through representation, "
            "decision-making, planning, action, interaction, and learning. "
            "Explain the purpose of each layer and how the layers work "
            "together to enable autonomous behavior."
        ),
        "expected": "answer",
    },
    {
        "id": 6,
        "name": "Single-agent vs Multi-Agent",
        "query": (
            "How does a Multi-Agent System differ from a single-agent "
            "system in terms of task decomposition, parallel execution, "
            "feedback, specialization, and coordination? Use the "
            "manufacturing examples from the document to explain the difference."
        ),
        "expected": "answer",
    },
    {
        "id": 7,
        "name": "Orchestration",
        "query": (
            "Explain how an orchestrator coordinates multiple agents. "
            "How are tasks broken down and allocated to agents based on "
            "their capabilities, availability, workload, dependencies, "
            "and deadlines? Also explain how task execution and verification "
            "fit into this process."
        ),
        "expected": "answer",
    },
    {
        "id": 8,
        "name": "Execution patterns",
        "query": (
            "What are the sequential, parallel, and iterative execution "
            "patterns described for multi-agent workflows? Explain when "
            "each pattern is appropriate and how the choice of execution "
            "pattern affects workflow efficiency."
        ),
        "expected": "answer",
    },
    {
        "id": 9,
        "name": "Verification and learning",
        "query": (
            "Explain the relationship between verification, execution, "
            "feedback, and learning in a multi-agent system. Why is "
            "verification important for reliability, and how can feedback "
            "from verification improve future task execution?"
        ),
        "expected": "answer",
    },
    {
        "id": 10,
        "name": "Challenges and mitigation",
        "query": (
            "What are the major challenges involved in designing and "
            "deploying Multi-Agent Systems according to the document? "
            "Identify each challenge and explain the corresponding "
            "mitigation strategy proposed for it."
        ),
        "expected": "answer",
    },
    {
        "id": 11,
        "name": "AI category comparison",
        "query": (
            "Compare Traditional AI, Non-agentic AI, Generative AI, "
            "and Agentic AI based on their definitions, capabilities, "
            "autonomy, and ability to perform actions."
        ),
        "expected": "answer",
    },
    {
        "id": 12,
        "name": "Industry applications",
        "query": (
            "What industries and practical application areas for "
            "Agentic AI are discussed in the document? Explain how "
            "Agentic AI can be applied across those areas."
        ),
        "expected": "answer",
    },
    {
        "id": 13,
        "name": "Why Multi-Agent Systems",
        "query": (
            "Based only on the document, why might a Multi-Agent System "
            "be preferred over a single-agent system for a complex "
            "manufacturing workflow? Explain using the concepts of "
            "specialization, parallel execution, feedback, and task allocation."
        ),
        "expected": "answer",
    },
    {
        "id": 14,
        "name": "Agent communication",
        "query": (
            "According to the document, what are the consequences of "
            "having too much communication between agents and what are "
            "the consequences of having too little communication? "
            "Why is the balance important?"
        ),
        "expected": "answer",
    },
    {
        "id": 15,
        "name": "False-premise hallucination",
        "query": (
            "According to the document, what are the seven core pillars "
            "of Agentic AI? Explain the seventh pillar in detail."
        ),
        "expected": "refusal",
    },
    {
        "id": 16,
        "name": "Unsupported factual claim",
        "query": (
            "According to the document, which company currently has the "
            "world's most advanced Agentic AI system, and what evidence "
            "does the document provide for that claim?"
        ),
        "expected": "refusal",
    },
    {
        "id": 17,
        "name": "Out-of-domain question",
        "query": (
            "What is the capital of France, and how does the document's "
            "discussion of Agentic AI relate to the French government's "
            "AI strategy?"
        ),
        "expected": "refusal",
    },
    {
        "id": 18,
        "name": "Deep multi-hop RAG",
        "query": (
            "Using only the document, explain how the six core pillars "
            "of Agentic AI relate to the layered architecture of a "
            "Multi-Agent System. Then explain how planning, task allocation, "
            "execution, verification, and learning work together when "
            "multiple specialized agents collaborate. Finally, identify "
            "the major challenges introduced by this architecture and the "
            "mitigation strategies proposed by the document."
        ),
        "expected": "answer",
    },
    {
        "id": 19,
        "name": "Representation vs perception",
        "query": (
            "What is the role of the representation layer in the "
            "Multi-Agent System architecture, and how is it different "
            "from the perception layer?"
        ),
        "expected": "answer",
    },
    {
        "id": 20,
        "name": "Execution dependency",
        "query": (
            "What are the three execution patterns described for "
            "multi-agent workflows, and what type of dependency makes "
            "sequential execution appropriate?"
        ),
        "expected": "answer",
    },
    {
        "id": 21,
        "name": "Core pillars and architecture",
        "query": (
            "How do perception, reasoning, planning, execution, and "
            "verification contribute to autonomous decision-making, "
            "and how do these concepts map onto the different "
            "architectural layers described later in the document?"
        ),
        "expected": "answer",
    },
    {
        "id": 22,
        "name": "Multi-agent reliability",
        "query": (
            "How does the document propose maintaining reliability "
            "when different agents perform different tasks? Discuss "
            "task allocation, verification, predefined success criteria, "
            "feedback loops, and human oversight."
        ),
        "expected": "answer",
    },
    {
        "id": 23,
        "name": "Complex system challenges",
        "query": (
            "If an organization wanted to deploy a large Multi-Agent "
            "System, what technical and organizational difficulties "
            "would it need to consider according to the document? "
            "Group the challenges into system design, integration, "
            "security, coordination, scalability, cost, development, "
            "and orchestration concerns, and explain the mitigation strategies."
        ),
        "expected": "answer",
    },
    {
        "id": 24,
        "name": "Manufacturing scenario",
        "query": (
            "Imagine the manufacturing example described in the document "
            "where multiple agents handle quality inspection, machinery "
            "monitoring, and production scheduling. Explain how these "
            "agents could collaborate, why parallel execution could be "
            "useful, how their outputs could be verified, and how feedback "
            "could improve the system over time."
        ),
        "expected": "answer",
    },
    {
        "id": 25,
        "name": "Pillars vs structural layers",
        "query": (
            "What is the difference between the core pillars of Agentic AI "
            "and the structural layers of a Multi-Agent System? Explain "
            "why these should not be treated as exactly the same "
            "architectural concept."
        ),
        "expected": "answer",
    },
    {
        "id": 26,
        "name": "Communication and orchestration",
        "query": (
            "Why is communication considered important in a Multi-Agent "
            "System? Explain the problems caused by excessive communication "
            "and insufficient communication, and describe how orchestration "
            "helps coordinate agents."
        ),
        "expected": "answer",
    },
    {
        "id": 27,
        "name": "End-to-end workflow",
        "query": (
            "Trace an Agentic AI workflow from receiving information "
            "from the environment to producing an action and learning "
            "from the result. Explain where perception, representation, "
            "reasoning, planning, execution, verification, interaction, "
            "and learning occur in this process."
        ),
        "expected": "answer",
    },
    {
        "id": 28,
        "name": "Architecture and challenges",
        "query": (
            "How do the architectural characteristics of Multi-Agent "
            "Systems contribute to challenges such as interoperability, "
            "conflict resolution, scalability, data security, and "
            "sophisticated orchestration? For each challenge, explain "
            "the mitigation strategy given in the document."
        ),
        "expected": "answer",
    },
    {
        "id": 29,
        "name": "Grounded reasoning",
        "query": (
            "According to the document, why can Multi-Agent Systems "
            "improve efficiency compared with a single-agent approach, "
            "and what new problems can be introduced by having multiple "
            "autonomous agents?"
        ),
        "expected": "answer",
    },
    {
        "id": 30,
        "name": "Complete RAG stress test",
        "query": (
            "Provide a comprehensive explanation of Agentic AI based "
            "only on the document. Start with its definition and "
            "distinguish it from Traditional AI, Non-agentic AI, "
            "Generative AI, and LLMs. Then explain the six core pillars "
            "and the layered architecture of an agentic system. Next, "
            "explain how single-agent and Multi-Agent Systems differ, "
            "including task decomposition, communication, orchestration, "
            "execution patterns, verification, and learning. Finally, "
            "discuss the major challenges of Multi-Agent Systems and "
            "the corresponding mitigation strategies described in "
            "the document."
        ),
        "expected": "answer",
    },
]



def run_test(test: dict) -> bool:
    """Run one RAG test and print its result."""

    print("\n" + "=" * 80)
    print(f"TEST {test['id']}: {test['name']}")
    print("=" * 80)

    print(f"\nQuestion:\n{test['query']}")

    result = rag_graph.invoke(
        {
            "question": test["query"],
            "context": [],
            "retrieved_context_chunks": [],
            "answer": "",
            "score": 0.0,
        }
    )

    answer = result.get("answer", "")
    chunks = result.get("retrieved_context_chunks", [])
    confidence = result.get("score", 0.0)

    print(f"\nAnswer:\n{answer}")

    print(f"\nConfidence: {confidence}")

    print("\nRetrieved chunks:")

    if not chunks:
        print("  No chunks retrieved.")
    else:
        for index, chunk in enumerate(chunks, start=1):
            print(
                f"  {index}. "
                f"page={chunk.get('page')} "
                f"score={chunk.get('score')}"
            )

    if test["expected"] == "refusal":
        passed = answer == (
            "I cannot answer based on the provided document."
        )
    else:
        passed = bool(answer) and answer != (
            "I cannot answer based on the provided document."
        )

    print(
        f"\nResult: {'PASS' if passed else 'FAIL'}"
    )

    return passed


def main():
    """Run the complete RAG test suite."""

    passed = 0
    failed = 0

    print("\n" + "#" * 80)
    print("AGENTIC AI RAG TEST SUITE")
    print("#" * 80)

    for test in TEST_QUERIES:
        try:
            if run_test(test):
                passed += 1
            else:
                failed += 1

            # Small delay to reduce Groq rate-limit pressure.
            time.sleep(1)

        except Exception as exc:
            failed += 1

            print("\nResult: ERROR")
            print(f"Error: {exc}")

            time.sleep(1)

    total = passed + failed

    print("\n" + "#" * 80)
    print("TEST SUMMARY")
    print("#" * 80)

    print(f"Total : {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nAll tests passed.")
    else:
        print(
            f"\n{failed} test(s) failed. "
            "Inspect the retrieved chunks and confidence scores."
        )


if __name__ == "__main__":
    main()
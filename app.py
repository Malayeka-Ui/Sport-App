# ==============================================================================
# MODULE 2: AI RULES ARBITER & TACTICAL COACH (POWERED BY GROQ)
# ==============================================================================
elif nav_choice == "🤖 AI Rules & Tactical Coach":
    st.header("🤖 AI Rules Arbiter & Tactical Coach")
    st.caption("Powered by Groq LPU Ultra-Low Latency Inference & BWF Regulations Engine")

    # Read key from secrets if present
    default_groq_key = st.secrets.get("GROQ_API_KEY", "")

    with st.expander("🔑 Configure Groq API Settings"):
        groq_api_key = st.text_input(
            "Groq API Key (starts with gsk_):",
            value=default_groq_key,
            type="password",
            help="Free key from console.groq.com. If omitted, the app uses built-in BWF rules.",
        )
        model_choice = st.selectbox(
            "Select Inference Model:",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0,
            help="70B offers the deepest tactical reasoning; 8B offers maximum generation speed.",
        )

    # Built-in fallback rule engine
    RULE_KNOWLEDGE_BASE = {
        "service height": "Under BWF Law 9.1.6, the whole shuttle must be below 1.15 meters (3.77 feet) from the court surface at the instant of racket impact.",
        "fault": "Common BWF faults include: shuttle landing outside lines, shuttle passing through or under the net, player/racket touching the net during play, hitting the shuttle before it crosses the net, and double-hitting.",
        "let": "A 'Let' halts play and replays the rally: called when an unforeseen accident occurs, shuttle disintegrates, receiver is not ready, or a stray shuttlecock enters the court.",
        "interval": "A 60-second break occurs when the leading score reaches 11 points in any game. A 120-second (2-minute) break occurs between consecutive games.",
        "doubles positioning": "Attacking setup: Front-and-back formation (rear player hits steep smashes/drops, front player kills net returns). Defensive setup: Side-by-side midcourt to cover all angles.",
        "singles positioning": "Recover to the central T base (1-2 steps behind the service line intersection) using a synchronized split-step at the opponent's point of impact.",
    }

    def match_internal_rules(query_lower):
        for key, answer in RULE_KNOWLEDGE_BASE.items():
            if key in query_lower:
                return answer
        if "serve" in query_lower:
            return "BWF Law 9: Both feet must stay stationary inside the service court. The racket must hit the shuttle base below 1.15m in an uninterrupted forward stroke."
        elif "net" in query_lower:
            return "BWF Law 13.4: Touching the net with your racket, body, or clothing while the shuttle is live is an immediate fault."
        return (
            "**BWF Rule Engine (Offline Mode):**\n"
            "Ask about *'service height'*, *'touching net'*, *'let calls'*, *'intervals'*, or *'doubles positioning'*. "
            "To unlock open-ended tactics and natural conversation, add your Groq API key in the panel above."
        )

    def get_groq_coach_response(user_query, api_key, model_name):
        if api_key:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                
                system_prompt = (
                    "You are an elite BWF Certified High-Performance Badminton Coach and International Umpire. "
                    "Provide clear, tactical, rules-accurate, and concise advice. "
                    "Address footwork, racket tension, singles/doubles strategy, or BWF laws authoritatively."
                )
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    model=model_name,
                    temperature=0.3,
                    max_tokens=600,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                return f"*(Groq API notice: {e}. Defaulting to offline knowledge base)*\n\n" + match_internal_rules(user_query.lower())

        return match_internal_rules(user_query.lower())

    # Display chat log
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User chat input
    user_input = st.chat_input("Ask a question about rules, tactics, shot selection, or drills...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("AI Coach analyzing with Groq..."):
            response_text = get_groq_coach_response(user_input, groq_api_key, model_choice)

        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

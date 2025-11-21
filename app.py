    import streamlit as st
    import pandas as pd
    from datetime import datetime
    import plotly.express as px # Added Plotly for enhanced charting

    # --- PYTHON SERVER-SIDE SENTIMENT SIMULATION ---

    def simulate_sentiment(message):
        """
        A simple Python function to simulate sentiment analysis, returning a mood score (0-100).
        In a production application, this would use a robust NLP model (e.g., using libraries like 
        NLTK, spaCy, or calling a large language model API).
        """
        positive_words = ['happy', 'great', 'good', 'better', 'joy', 'love', 'excited', 'amazing', 'thankful', 'relief', 'got through']
        negative_words = ['sad', 'stressed', 'anxious', 'tired', 'bad', 'hard', 'depressed', 'fear', 'alone', 'overwhelmed', 'difficult', 'struggling']
        
        score = 50 # Start at neutral
        
        words = message.lower().split()
        sentiment_delta = 0

        for word in words:
            # Check for positive/negative sentiment
            if word in positive_words:
                sentiment_delta += 8
            elif word in negative_words:
                sentiment_delta -= 10
                
        # Simple adjustment for message length impact
        length_bonus = min(len(message) // 50 * 5, 15)
        sentiment_delta += length_bonus
        
        score = min(100, max(0, score + sentiment_delta))
        return round(score)

    # --- STREAMLIT APP UTILITIES ---

    def get_mood_label(score):
        """Assigns a descriptive label based on the numerical score."""
        if score >= 70:
            return 'Positive'
        elif score >= 40:
            return 'Mixed/Neutral'
        else:
            return 'Stressed/Low'

    def init_session_state():
        """Initializes the data and chat history in Streamlit's session state."""
        # Initialize DataFrame for chart data if not present
        if 'chat_entries' not in st.session_state:
            st.session_state.chat_entries = pd.DataFrame(
                columns=['timestamp', 'message', 'moodScore']
            )
        # Initialize chat history list if not present
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = [
                {'role': 'AI', 'content': "Hello! I'm running on a Streamlit server with Pandas. Share what's on your mind. I'll analyze your sentiment and plot your score on the chart!"}
            ]

    def handle_checkin():
        """Processes the user's message, calculates the score, updates data, and clears the input."""
        user_message = st.session_state.message_input
        
        if user_message:
            # 1. Run AI simulation
            mood_score = simulate_sentiment(user_message)
            current_time = datetime.now()

            # 2. Create the new entry for the DataFrame
            new_entry_df = pd.DataFrame([{
                'timestamp': current_time, 
                'message': user_message, 
                'moodScore': mood_score
            }])

            # 3. Append to Pandas DataFrame (Mood data for charting)
            st.session_state.chat_entries = pd.concat([st.session_state.chat_entries, new_entry_df], ignore_index=True)

            # 4. Append to Chat History
            st.session_state.chat_history.append({'role': 'User', 'content': user_message})
            st.session_state.chat_history.append({
                'role': 'AI', 
                'content': f"I hear you. My analysis shows a **Mood Score of {mood_score}**. I detected signals of **{get_mood_label(mood_score)}**. Remember to take a moment for yourself."
            })
            
            # 5. Clear input field via key reference
            st.session_state.message_input = ""

    def analyze_daily_stats(df):
        """Calculates and displays daily statistical summaries using Pandas."""
        st.subheader("Daily Stats & Aggregations")
        
        if df.empty:
            st.info("No data entries yet to calculate daily statistics.")
            return

        # Group by date (Pandas operation)
        df['date'] = df['timestamp'].dt.date
        daily_summary = df.groupby('date')['moodScore'].agg(
            avg_mood='mean',
            total_entries='count',
            max_mood='max',
            min_mood='min'
        ).reset_index()
        
        # Filter for today's data (assuming all entries are recent for this demo)
        today = datetime.now().date()
        today_stats = daily_summary[daily_summary['date'] == today].iloc[0] if not daily_summary[daily_summary['date'] == today].empty else None
        
        if today_stats is not None:
            
            avg_mood = round(today_stats['avg_mood'])
            total_entries = int(today_stats['total_entries'])
            
            # Display key metrics using Streamlit metric cards
            col_avg, col_count, col_min_max = st.columns(3)
            
            with col_avg:
                st.metric(
                    label="Today's Average Mood Score", 
                    value=f"{avg_mood} ({get_mood_label(avg_mood)})", 
                    delta_color="off" # Static label, no delta needed
                )
            
            with col_count:
                st.metric(
                    label="Total Check-ins Today", 
                    value=total_entries
                )
            
            with col_min_max:
                st.metric(
                    label="Score Range (Min/Max)", 
                    value=f"{int(today_stats['min_mood'])} / {int(today_stats['max_mood'])}"
                )

    # --- MAIN STREAMLIT APPLICATION ---

    def main_app():
        # Set page configuration for wide layout
        st.set_page_config(layout="wide", page_title="AI Mental Health Monitor")
        init_session_state()

        st.title("🧠 AI Mental Health Monitor")

        # Layout: Two main columns (Chat/Input and Monitoring/Analysis)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Daily Check-In Chat")
            
            # Chat Display Area
            chat_container = st.container(border=True, height=500)
            with chat_container:
                for chat in st.session_state.chat_history:
                    if chat['role'] == 'AI':
                        st.markdown(f"""
                            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                                <div style="background-color: #e0f2fe; padding: 10px 15px; border-radius: 12px; border-top-left-radius: 2px; max-width: 80%; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);">
                                    <p style="font-weight: bold; color: #0284c7; margin: 0; font-size: 0.9em;">Mind Monitor AI</p>
                                    <p style="margin: 5px 0 0 0; font-size: 0.9em;">{chat['content']}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        # User message
                        st.markdown(f"""
                            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                <div style="background-color: #4f46e5; color: white; padding: 10px 15px; border-radius: 12px; border-top-right-radius: 2px; max-width: 80%; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);">
                                    <p style="font-weight: bold; margin: 0; font-size: 0.9em;">You</p>
                                    <p style="margin: 5px 0 0 0; font-size: 0.9em;">{chat['content']}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            # Input Area 
            st.text_input(
                "How are you feeling today?", 
                key="message_input", 
                on_change=handle_checkin, 
                placeholder="e.g., 'Today was stressful but I got through it.'",
                label_visibility="collapsed"
            )
            st.button("Send Check-in", on_click=handle_checkin, use_container_width=True)

        with col2:
            
            # --- 1. Daily Statistics Section ---
            analyze_daily_stats(st.session_state.chat_entries)
            
            st.markdown("---") # Separator

            # --- 2. Mood Score Trend Chart ---
            st.subheader("Mood Score Trend (Time Series)")

            df = st.session_state.chat_entries
            
            if not df.empty:
                # Prepare data for charting
                df_sorted = df.sort_values(by='timestamp', ascending=True)
                df_sorted['Time'] = df_sorted['timestamp'].dt.strftime('%H:%M:%S')
                
                # Use Plotly for an interactive line chart (better than native st.line_chart)
                fig = px.line(
                    df_sorted, 
                    x='Time', 
                    y='moodScore', 
                    markers=True,
                    line_shape='spline',
                    color_discrete_sequence=['#4f46e5'],
                    title='Mood Score Over Time Today'
                )
                fig.update_layout(
                    yaxis_range=[0, 100],
                    showlegend=False,
                    margin=dict(t=50, l=0, r=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Optional: Display the raw data for inspection
                with st.expander("View Raw Mood Data (Pandas DataFrame)"):
                    # Only display relevant columns
                    st.dataframe(
                        df_sorted[['Time', 'message', 'moodScore']].sort_values(by='Time', ascending=False), 
                        hide_index=True
                    )
            else:
                st.info("Enter your first check-in message to see the Monitoring Data.")


            # Footer / Disclaimer
            st.markdown("""
            <div style="margin-top: 20px; padding: 15px; background-color: #fffbe3; border: 1px solid #fde68a; border-radius: 8px; font-size: 0.875rem; color: #92400e;">
                <p style="font-weight: bold; margin-bottom: 5px;">About the Mood Score & Analysis:</p>
                <p>The score and daily statistics are calculated by the Python backend using a simple sentiment analysis simulation and Pandas aggregation. The data is temporarily stored in a Streamlit session-managed Pandas DataFrame.</p>
                <p style="margin-top: 5px; font-size: 0.75rem;">
                    *This is a demo for educational purposes and should not be used for real mental health diagnosis or treatment.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Run the app
    if __name__ == '__main__':
        # You will need to install Plotly in addition to the original requirements:
        # pip install streamlit pandas plotly
        # Then run: streamlit run mood_monitor_app.py
        main_app()
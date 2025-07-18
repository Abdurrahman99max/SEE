import os
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END

try:
    from langgraph.prebuilt.tool_executor import ToolExecutor
except ImportError:
    # Fallback manual implementation
    class ToolExecutor:
        def __init__(self, tools):
            self.tools = {tool.name: tool for tool in tools}
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel


# =============================================================================
# DATA MODELS AND STATE DEFINITION
# =============================================================================

class CityData(BaseModel):
    """Data model for city information"""
    name: str
    country: str
    weather: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    events: Optional[List[Dict[str, Any]]] = None
    cost_of_living: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    recommendation: Optional[str] = None


class AgentState(TypedDict):
    """State definition for the LangGraph agent"""
    messages: List[Any]
    user_request: str
    parsed_requirements: Dict[str, Any]
    candidate_cities: List[str]
    city_data: Dict[str, CityData]
    current_step: str
    errors: List[str]
    reflection_notes: List[str]
    final_report: Optional[str]
    iteration_count: int


# =============================================================================
# MOCK TOOL IMPLEMENTATIONS
# =============================================================================

@tool
def get_weather_data(city: str, country: str) -> Dict[str, Any]:
    """Get weather data for a city during summer months"""
    # Mock weather data - in production, this would call OpenWeatherMap API
    weather_patterns = {
        "Bangkok": {"temp": 32, "humidity": 75, "rainfall": 250, "condition": "Hot & Humid"},
        "Prague": {"temp": 24, "humidity": 60, "rainfall": 80, "condition": "Pleasant"},
        "Lisbon": {"temp": 28, "humidity": 65, "rainfall": 20, "condition": "Warm & Sunny"},
        "Budapest": {"temp": 26, "humidity": 55, "rainfall": 60, "condition": "Warm"},
        "Warsaw": {"temp": 23, "humidity": 65, "rainfall": 70, "condition": "Mild"},
        "Mexico City": {"temp": 22, "humidity": 70, "rainfall": 120, "condition": "Mild & Rainy"},
        "Krakow": {"temp": 24, "humidity": 60, "rainfall": 90, "condition": "Pleasant"},
        "Istanbul": {"temp": 29, "humidity": 70, "rainfall": 30, "condition": "Warm"}
    }

    base_data = weather_patterns.get(city, {
        "temp": random.randint(20, 35),
        "humidity": random.randint(50, 80),
        "rainfall": random.randint(20, 200),
        "condition": "Variable"
    })

    return {
        "city": city,
        "country": country,
        "summer_avg_temp": base_data["temp"],
        "humidity": base_data["humidity"],
        "rainfall_mm": base_data["rainfall"],
        "condition": base_data["condition"],
        "uv_index": random.randint(6, 10)
    }


@tool
def get_safety_index(city: str, country: str) -> Dict[str, Any]:
    """Get safety index and crime data for a city"""
    # Mock safety data - in production, this would call a real safety API
    safety_scores = {
        "Bangkok": {"index": 65, "crime": "moderate"},
        "Prague": {"index": 85, "crime": "low"},
        "Lisbon": {"index": 82, "crime": "low"},
        "Budapest": {"index": 75, "crime": "low-moderate"},
        "Warsaw": {"index": 78, "crime": "low"},
        "Mexico City": {"index": 45, "crime": "high"},
        "Krakow": {"index": 88, "crime": "very low"},
        "Istanbul": {"index": 60, "crime": "moderate"}
    }

    # Simulate occasional API failures for reflection testing
    if random.random() < 0.1:  # 10% chance of failure
        return {"error": f"Safety data temporarily unavailable for {city}"}

    base_data = safety_scores.get(city, {
        "index": random.randint(40, 90),
        "crime": "unknown"
    })

    return {
        "city": city,
        "country": country,
        "safety_index": base_data["index"],
        "crime_level": base_data["crime"],
        "tourist_safety": "good" if base_data["index"] > 70 else "moderate",
        "local_tips": f"Standard precautions recommended for {city}"
    }


@tool
def get_local_events(city: str, country: str, month: str = "July") -> List[Dict[str, Any]]:
    """Get local events and festivals for summer period"""
    # Mock event data - in production, this would call Eventbrite or similar API
    event_templates = [
        {"type": "music", "name": "Summer Music Festival", "price": "€25-50"},
        {"type": "cultural", "name": "Cultural Heritage Days", "price": "Free"},
        {"type": "food", "name": "Street Food Market", "price": "€5-15"},
        {"type": "art", "name": "Art Gallery Exhibition", "price": "€8-12"},
        {"type": "outdoor", "name": "Outdoor Cinema", "price": "€10-20"}
    ]

    num_events = random.randint(2, 4)
    events = []

    for i in range(num_events):
        template = random.choice(event_templates)
        events.append({
            "name": f"{city} {template['name']}",
            "type": template["type"],
            "date": f"{month} {random.randint(1, 30)}, 2024",
            "price": template["price"],
            "description": f"Popular {template['type']} event in {city}"
        })

    return events


@tool
def get_cost_of_living(city: str, country: str) -> Dict[str, Any]:
    """Get cost of living data including daily expenses"""
    # Mock cost data - in production, this would call Numbeo API
    cost_data = {
        "Bangkok": {"accommodation": 25, "food": 15, "transport": 5, "activities": 20},
        "Prague": {"accommodation": 40, "food": 20, "transport": 8, "activities": 25},
        "Lisbon": {"accommodation": 45, "food": 25, "transport": 10, "activities": 30},
        "Budapest": {"accommodation": 35, "food": 18, "transport": 6, "activities": 22},
        "Warsaw": {"accommodation": 38, "food": 20, "transport": 7, "activities": 25},
        "Mexico City": {"accommodation": 30, "food": 12, "transport": 4, "activities": 18},
        "Krakow": {"accommodation": 32, "food": 16, "transport": 5, "activities": 20},
        "Istanbul": {"accommodation": 28, "food": 14, "transport": 3, "activities": 16}
    }

    base_costs = cost_data.get(city, {
        "accommodation": random.randint(25, 60),
        "food": random.randint(10, 30),
        "transport": random.randint(3, 15),
        "activities": random.randint(15, 35)
    })

    daily_total = sum(base_costs.values())

    return {
        "city": city,
        "country": country,
        "daily_costs": base_costs,
        "daily_total": daily_total,
        "currency": "USD",
        "affordability": "budget" if daily_total < 50 else "moderate" if daily_total < 80 else "expensive"
    }


# =============================================================================
# AGENT NODES (COGNITIVE FUNCTIONS)
# =============================================================================

class TravelAgent:
    """Autonomous travel recommendation agent with LangGraph"""

    def __init__(self, groq_api_key: str = None):
        # API key handling with multiple sources
        if groq_api_key is None:
            groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key or groq_api_key == "your-groq-api-key-here":
            raise ValueError(
                "Please provide a valid Groq API key. You can:\n"
                "1. Pass it directly: TravelAgent(groq_api_key='your-key')\n"
                "2. Set environment variable: export GROQ_API_KEY='your-key'\n"
                "3. Get your key from: https://console.groq.com/keys"
            )

        try:
            self.llm = ChatGroq(
                model="llama3-70b-8192",  # Groq's fast Llama 3 model
                temperature=0.1,
                groq_api_key=groq_api_key,
                max_tokens=1000
            )
            # Test the connection
            test_response = self.llm.invoke([HumanMessage(content="Test")])
            print("✅ Groq API connection successful!")
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq API: {str(e)}")

        self.tools = [get_weather_data, get_safety_index, get_local_events, get_cost_of_living]
        # Note: We'll call tools directly instead of using ToolExecutor

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes for each cognitive step
        workflow.add_node("parse_requirements", self._parse_requirements)
        workflow.add_node("select_candidate_cities", self._select_candidate_cities)
        workflow.add_node("gather_city_data", self._gather_city_data)
        workflow.add_node("reflect_and_validate", self._reflect_and_validate)
        workflow.add_node("analyze_and_rank", self._analyze_and_rank)
        workflow.add_node("generate_report", self._generate_report)

        # Define the workflow edges
        workflow.set_entry_point("parse_requirements")
        workflow.add_edge("parse_requirements", "select_candidate_cities")
        workflow.add_edge("select_candidate_cities", "gather_city_data")
        workflow.add_edge("gather_city_data", "reflect_and_validate")

        # Conditional edge based on reflection
        workflow.add_conditional_edges(
            "reflect_and_validate",
            self._should_continue_or_retry,
            {
                "continue": "analyze_and_rank",
                "retry": "gather_city_data",
                "finalize": "generate_report"
            }
        )

        workflow.add_edge("analyze_and_rank", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def _parse_requirements(self, state: AgentState) -> AgentState:
        """Parse and decompose the user request into structured requirements"""
        print("🧠 Step 1: Parsing requirements and decomposing goals...")

        prompt = f"""
        Analyze this user request: "{state['user_request']}"

        Extract the following requirements:
        1. Travel season/timing
        2. Budget level (affordable = what daily budget?)
        3. Required information types
        4. Geographic preferences
        5. Number of cities to recommend

        Return a structured analysis in JSON format.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Parse the LLM response (simplified for demo)
            requirements = {
                "season": "summer",
                "budget_level": "affordable",
                "max_daily_cost": 70,  # USD
                "required_info": ["weather", "safety", "events", "cost"],
                "num_cities": 5,
                "geographic_scope": "international"
            }

            state["parsed_requirements"] = requirements
            state["current_step"] = "requirements_parsed"
            state["messages"].append(AIMessage(content=f"Parsed requirements: {requirements}"))

        except Exception as e:
            error_msg = f"Failed to parse requirements: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            # Fallback to default requirements
            state["parsed_requirements"] = {
                "season": "summer",
                "budget_level": "affordable",
                "max_daily_cost": 70,
                "required_info": ["weather", "safety", "events", "cost"],
                "num_cities": 5,
                "geographic_scope": "international"
            }

        return state

    def _select_candidate_cities(self, state: AgentState) -> AgentState:
        """Select candidate cities based on parsed requirements"""
        print("🗺️ Step 2: Selecting candidate cities...")

        # Curated list of affordable summer destinations
        affordable_cities = [
            "Bangkok, Thailand",
            "Prague, Czech Republic",
            "Lisbon, Portugal",
            "Budapest, Hungary",
            "Warsaw, Poland",
            "Mexico City, Mexico",
            "Krakow, Poland",
            "Istanbul, Turkey"
        ]

        # Select cities based on requirements
        num_cities = state["parsed_requirements"]["num_cities"]
        selected_cities = affordable_cities[:num_cities]

        state["candidate_cities"] = selected_cities
        state["current_step"] = "cities_selected"
        state["messages"].append(AIMessage(content=f"Selected {len(selected_cities)} candidate cities"))

        return state

    def _gather_city_data(self, state: AgentState) -> AgentState:
        """Gather data for each city using available tools"""
        print("🔍 Step 3: Gathering comprehensive city data...")

        for city_str in state["candidate_cities"]:
            city_name, country = city_str.split(", ")

            print(f"  → Collecting data for {city_name}, {country}")

            # Initialize city data object
            city_data = CityData(name=city_name, country=country)

            # Gather weather data
            try:
                weather_data = get_weather_data.invoke({"city": city_name, "country": country})
                city_data.weather = weather_data
            except Exception as e:
                state["errors"].append(f"Weather data failed for {city_name}: {str(e)}")

            # Gather safety data
            try:
                safety_data = get_safety_index.invoke({"city": city_name, "country": country})
                if "error" not in safety_data:
                    city_data.safety = safety_data
                else:
                    state["errors"].append(f"Safety data unavailable for {city_name}")
            except Exception as e:
                state["errors"].append(f"Safety data failed for {city_name}: {str(e)}")

            # Gather events data
            try:
                events_data = get_local_events.invoke({"city": city_name, "country": country})
                city_data.events = events_data
            except Exception as e:
                state["errors"].append(f"Events data failed for {city_name}: {str(e)}")

            # Gather cost data
            try:
                cost_data = get_cost_of_living.invoke({"city": city_name, "country": country})
                city_data.cost_of_living = cost_data
            except Exception as e:
                state["errors"].append(f"Cost data failed for {city_name}: {str(e)}")

            state["city_data"][city_str] = city_data

        state["current_step"] = "data_gathered"
        state["iteration_count"] = state.get("iteration_count", 0) + 1

        return state

    def _reflect_and_validate(self, state: AgentState) -> AgentState:
        """Reflect on data quality and decide whether to continue or retry"""
        print("🤔 Step 4: Reflecting on data quality and completeness...")

        reflection_prompt = f"""
           Analyze the data collection results:

           Cities processed: {len(state['city_data'])}
           Errors encountered: {len(state['errors'])}
           Current iteration: {state.get('iteration_count', 1)}

           Errors: {state['errors']}

           Questions to consider:
           1. Do I have sufficient data to make good recommendations?
           2. Are there critical gaps that need addressing?
           3. Should I retry failed data collection or proceed with available data?
           4. Is the data quality acceptable for the user's needs?

           Provide a decision: CONTINUE, RETRY, or FINALIZE
           """

        response = self.llm.invoke([HumanMessage(content=reflection_prompt)])
        decision = response.content.strip().upper()

        # Analyze data completeness
        complete_cities = 0
        for city_data in state["city_data"].values():
            if (city_data.weather and city_data.safety and
                    city_data.events and city_data.cost_of_living):
                complete_cities += 1

        # Reflection logic
        if complete_cities >= 3 and len(state["errors"]) < 5:
            reflection_decision = "CONTINUE"
        elif state.get("iteration_count", 0) >= 2:
            reflection_decision = "FINALIZE"  # Prevent infinite loops
        else:
            reflection_decision = "RETRY"
        iteration_limit = 3
        current_iter = state.get("iteration_count", 1)

        if decision == "RETRY" and current_iter >= iteration_limit:
            decision = "FINALIZE"
        if decision not in {"CONTINUE", "RETRY", "FINALIZE"}:
            if complete_cities >= 3 and len(state["errors"]) < 5:
                decision = "CONTINUE"
            elif current_iter >= iteration_limit:
                decision = "FINALIZE"
            else:
                decision = "RETRY"
        else:
            # LLM RETRY kararı verdiyse bile iteration sayısını kontrol et
            if decision == "RETRY" and current_iter >= iteration_limit:
                decision = "FINALIZE"

        reflection_note = f"""
           Reflection Analysis:
           - Complete city profiles: {complete_cities}/{len(state['city_data'])}
           - Data collection errors: {len(state['errors'])}
           - Decision: {reflection_decision}
           - Reasoning: {'Sufficient data quality' if reflection_decision == 'CONTINUE' else 'Need more data' if reflection_decision == 'RETRY' else 'Proceeding with available data'}
           """

        state["reflection_notes"].append(reflection_note)
        state["current_step"] = f"reflected_{reflection_decision.lower()}"

        return state

    def _should_continue_or_retry(self, state: AgentState) -> str:
        """Conditional logic for reflection decision"""
        if "reflected_continue" in state["current_step"]:
            return "continue"
        elif "reflected_retry" in state["current_step"]:
            return "retry"
        else:
            return "finalize"

    def _analyze_and_rank(self, state: AgentState) -> AgentState:
        """Analyze collected data and rank cities"""
        print("📊 Step 5: Analyzing and ranking cities...")

        budget_limit = state["parsed_requirements"]["max_daily_cost"]

        for city_str, city_data in state["city_data"].items():
            score = 0
            factors = []

            # Cost factor (40% weight)
            if city_data.cost_of_living:
                daily_cost = city_data.cost_of_living["daily_total"]
                if daily_cost <= budget_limit:
                    cost_score = max(0, (budget_limit - daily_cost) / budget_limit * 40)
                    score += cost_score
                    factors.append(f"Cost: ${daily_cost}/day (score: {cost_score:.1f})")

            # Weather factor (20% weight)
            if city_data.weather:
                temp = city_data.weather["summer_avg_temp"]
                weather_score = max(0, 20 - abs(temp - 25) * 2)  # Prefer ~25°C
                score += weather_score
                factors.append(f"Weather: {temp}°C (score: {weather_score:.1f})")

            # Safety factor (30% weight)
            if city_data.safety:
                safety_index = city_data.safety["safety_index"]
                safety_score = (safety_index / 100) * 30
                score += safety_score
                factors.append(f"Safety: {safety_index}/100 (score: {safety_score:.1f})")

            # Events factor (10% weight)
            if city_data.events:
                events_score = min(len(city_data.events) * 2.5, 10)
                score += events_score
                factors.append(f"Events: {len(city_data.events)} (score: {events_score:.1f})")

            city_data.score = score

            # Generate recommendation text
            city_data.recommendation = self._generate_city_recommendation(city_data, factors)

        state["current_step"] = "analyzed_and_ranked"

        return state

    def _generate_city_recommendation(self, city_data: CityData, factors: List[str]) -> str:
        """Generate a recommendation paragraph for a city"""
        cost_info = f"${city_data.cost_of_living['daily_total']}/day" if city_data.cost_of_living else "cost unknown"
        weather_info = f"{city_data.weather['summer_avg_temp']}°C, {city_data.weather['condition']}" if city_data.weather else "weather unknown"
        safety_info = f"safety index {city_data.safety['safety_index']}/100" if city_data.safety else "safety unknown"
        events_count = len(city_data.events) if city_data.events else 0

        return f"""
        {city_data.name}, {city_data.country} offers excellent value for summer travel with {cost_info} average daily costs. 
        The weather is {weather_info}, making it comfortable for sightseeing. 
        With a {safety_info}, it's reasonably safe for tourists. 
        The city features {events_count} notable summer events and activities. 
        Overall score: {city_data.score:.1f}/100 based on cost-effectiveness, weather, safety, and cultural offerings.
        """

    def _generate_report(self, state: AgentState) -> AgentState:
        """Generate the final structured report"""
        print("📝 Step 6: Generating final report...")

        try:
            # Sort cities by score
            sorted_cities = sorted(
                state["city_data"].items(),
                key=lambda x: x[1].score or 0,
                reverse=True
            )

            # Generate HTML report with improved error handling
            html_report = self._create_html_report(sorted_cities, state)

            # Generate JSON report
            json_report = self._create_json_report(sorted_cities, state)

            state["final_report"] = html_report
            state["current_step"] = "report_generated"

        except Exception as e:
            error_msg = f"Failed to generate report: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

            # Fallback to simple text report
            state["final_report"] = self._create_simple_text_report(state["city_data"])
            state["current_step"] = "report_generated_fallback"

        return state

    def _create_html_report(self, sorted_cities: List, state: AgentState) -> str:
        """Create an HTML report with improved error handling"""
        try:
            # Fixed CSS styling - removed potential problematic characters
            css_styles = """
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #f5f5f5; 
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }
                .city { 
                    border: 1px solid #ddd; 
                    margin: 20px 0; 
                    padding: 15px; 
                    border-radius: 8px; 
                    background-color: #fafafa; 
                }
                .score { 
                    color: #2196F3; 
                    font-weight: bold; 
                    font-size: 18px; 
                }
                .cost { color: #4CAF50; }
                .weather { color: #FF9800; }
                .safety { color: #9C27B0; }
                .events { color: #607D8B; }
                .error { 
                    color: #F44336; 
                    font-size: 12px; 
                    background-color: #ffebee; 
                    padding: 5px; 
                    border-radius: 4px; 
                }
                h1 { color: #333; }
                h2 { color: #555; }
                h3 { color: #777; }
                .header-info { 
                    background-color: #e3f2fd; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-bottom: 20px; 
                }
            """

            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Summer Travel Recommendations</title>
                <style>{css_styles}</style>
            </head>
            <body>
                <div class="container">
                    <h1>🌍 Affordable Summer Travel Destinations</h1>
                    <div class="header-info">
                        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>Budget Target:</strong> Under ${state['parsed_requirements']['max_daily_cost']}/day</p>
                        <p><strong>Cities Analyzed:</strong> {len(state['city_data'])}</p>
                    </div>

                    <h2>🏆 Top Recommendations</h2>
            """

            # Add city information
            for i, (city_str, city_data) in enumerate(sorted_cities, 1):
                html_content += f"""
                <div class="city">
                    <h3>{i}. {city_data.name}, {city_data.country}</h3>
                    <div class="score">Overall Score: {city_data.score:.1f}/100</div>

                    <h4>💰 Cost Breakdown</h4>
                    {self._format_cost_html(city_data.cost_of_living)}

                    <h4>🌤️ Weather Information</h4>
                    {self._format_weather_html(city_data.weather)}

                    <h4>🛡️ Safety Details</h4>
                    {self._format_safety_html(city_data.safety)}

                    <h4>🎉 Local Events</h4>
                    {self._format_events_html(city_data.events)}

                    <h4>📋 Recommendation</h4>
                    <p>{city_data.recommendation.strip()}</p>
                </div>
                """

            # Add errors if any
            if state["errors"]:
                html_content += "<h2>⚠️ Data Collection Issues</h2><ul>"
                for error in state["errors"]:
                    html_content += f"<li class='error'>{error}</li>"
                html_content += "</ul>"

            # Add process summary
            html_content += """
                <h2>🔄 Process Summary</h2>
                <p>This report was generated through autonomous agent reasoning with the following steps:</p>
                <ol>
                    <li>Requirement parsing and goal decomposition</li>
                    <li>Candidate city selection based on affordability</li>
                    <li>Multi-tool data gathering (weather, safety, events, costs)</li>
                    <li>Reflection and quality validation</li>
                    <li>Multi-criteria analysis and ranking</li>
                    <li>Structured report generation</li>
                </ol>
                </div>
            </body>
            </html>
            """

            return html_content

        except Exception as e:
            print(f"❌ HTML generation failed: {str(e)}")
            return self._create_simple_text_report(state["city_data"])

    def _create_simple_text_report(self, city_data: Dict) -> str:
        """Create a simple text report as fallback"""
        report = "SUMMER TRAVEL RECOMMENDATIONS\n"
        report += "=" * 50 + "\n\n"

        for city_str, data in city_data.items():
            report += f"CITY: {data.name}, {data.country}\n"
            report += f"Score: {data.score:.1f}/100\n"

            if data.cost_of_living:
                report += f"Daily Cost: ${data.cost_of_living['daily_total']}\n"

            if data.weather:
                report += f"Weather: {data.weather['summer_avg_temp']}°C, {data.weather['condition']}\n"

            if data.safety:
                report += f"Safety: {data.safety['safety_index']}/100\n"

            report += f"Recommendation: {data.recommendation}\n"
            report += "-" * 30 + "\n\n"

        return report

    def _format_cost_html(self, cost_data: Dict) -> str:
        """Format cost data with error handling"""
        if not cost_data:
            return "<p>Cost data unavailable</p>"

        try:
            return (
                f"""
                <div class="cost">
                    <p><strong>Daily Total: ${cost_data['daily_total']}</strong></p>
                    <ul>
                        <li>Accommodation: ${cost_data['daily_costs']['accommodation']}</li>
                        <li>Food: ${cost_data['daily_costs']['food']}</li>
                        <li>Transport: ${cost_data['daily_costs']['transport']}</li>
                        <li>Activities: ${cost_data['daily_costs']['activities']}</li>
                    </ul>
                    <p>Affordability: {cost_data['affordability'].title()}</p>
                </div> 
                """
            )
        except Exception as e:
            return f"<p>Error formatting cost data: {str(e)}</p>"

    def _format_cost_html(self, cost_data: Dict) -> str:
        if not cost_data:
            return "<p>Cost data unavailable</p>"
        try:
            return f"""
            <div class="cost">
                <p><strong>Daily Total: ${cost_data['daily_total']}</strong></p>
                <ul>
                    <li>Accommodation: ${cost_data['daily_costs']['accommodation']}</li>
                    <li>Food: ${cost_data['daily_costs']['food']}</li>
                    <li>Transport: ${cost_data['daily_costs']['transport']}</li>
                    <li>Activities: ${cost_data['daily_costs']['activities']}</li>
                </ul>
                <p>Affordability: {cost_data['affordability'].title()}</p>
            </div>
            """
        except Exception as e:
            return f"<p>Error formatting cost data: {str(e)}</p>"

    def _format_weather_html(self, weather_data: Dict) -> str:
        if not weather_data:
            return "<p>Weather data unavailable</p>"

        return f"""
        <div class="weather">
            <p><strong>Condition:</strong> {weather_data['condition']}</p>
            <p><strong>Temperature:</strong> {weather_data['summer_avg_temp']}°C</p>
            <p><strong>Humidity:</strong> {weather_data['humidity']}%</p>
            <p><strong>Rainfall:</strong> {weather_data['rainfall_mm']}mm</p>
        </div>
        """

    def _format_safety_html(self, safety_data: Dict) -> str:
        if not safety_data:
            return "<p>Safety data unavailable</p>"

        return f"""
        <div class="safety">
            <p><strong>Safety Index:</strong> {safety_data['safety_index']}/100</p>
            <p><strong>Crime Level:</strong> {safety_data['crime_level'].title()}</p>
            <p><strong>Tourist Safety:</strong> {safety_data['tourist_safety'].title()}</p>
        </div>
        """

    def _format_events_html(self, events_data: List) -> str:
        if not events_data:
            return "<p>No events data available</p>"

        html = "<div class='events'><ul>"
        for event in events_data:
            html += f"<li><strong>{event['name']}</strong> - {event['type']} ({event['price']})</li>"
        html += "</ul></div>"

        return html

    def _create_json_report(self, sorted_cities: List, state: AgentState) -> Dict:
        """Create a JSON report for programmatic use"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "user_request": state["user_request"],
            "requirements": state["parsed_requirements"],
            "cities": [],
            "process_summary": {
                "steps_completed": ["parse", "select", "gather", "reflect", "analyze", "report"],
                "iterations": state.get("iteration_count", 1),
                "errors": state["errors"],
                "reflection_notes": state["reflection_notes"]
            }
        }

        for city_str, city_data in sorted_cities:
            city_report = {
                "rank": len(report["cities"]) + 1,
                "name": city_data.name,
                "country": city_data.country,
                "score": city_data.score,
                "data": {
                    "weather": city_data.weather,
                    "safety": city_data.safety,
                    "events": city_data.events,
                    "cost_of_living": city_data.cost_of_living
                },
                "recommendation": city_data.recommendation
            }
            report["cities"].append(city_report)

        return report

    def invoke(self, user_request: str) -> Dict[str, Any]:
        """Main entry point for the agent"""
        print(f"🚀 Starting autonomous travel agent for: '{user_request}'")

        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=user_request)],
            user_request=user_request,
            parsed_requirements={},
            candidate_cities=[],
            city_data={},
            current_step="starting",
            errors=[],
            reflection_notes=[],
            final_report=None,
            iteration_count=0
        )

        # Execute the graph
        result = self.graph.invoke(initial_state)

        print("✅ Agent execution completed!")

        return {
            "status": "success",
            "final_report": result["final_report"],
            "process_summary": {
                "steps": result["current_step"],
                "errors": result["errors"],
                "reflections": result["reflection_notes"],
                "cities_analyzed": len(result["city_data"])
            }
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Method 1: Direct API key
    agent = TravelAgent(groq_api_key="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")

# Method 2: Environment variable (recommended)
# First set: export GROQ_API_KEY="gsk_your_actual_groq_api_key_here"
# Then run without parameter

try:
    agent = TravelAgent()  # Will use environment variable

    # Invoke with the example request
    result = agent.invoke("Suggest affordable summer vacation cities with safety, weather, and cost details")

    # Display results
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(result["final_report"])

    # Save HTML report to file
    with open("travel_recommendations.html", "w") as f:
        f.write(result["final_report"])

    print("\n📄 Report saved to 'travel_recommendations.html'")
    print(f"🔄 Process completed with {len(result['process_summary']['errors'])} errors")

except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    print("\n🔑 How to get your Groq API key:")
    print("1. Go to https://console.groq.com/keys")
    print("2. Sign up or log in")
    print("3. Create a new API key")
    print("4. Set it as environment variable: export GROQ_API_KEY='your-key'")
    print("5. Or pass it directly to TravelAgent(groq_api_key='your-key')")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    print("Please check your API key and internet connection.")

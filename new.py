class Abdurrahman:
    def __init__(self):
        self.name = "Abdurrahman Onitilo"
        self.roles = ["Product Designer", "Nursing Student"]
        self.skills = [
            "UI/UX Design", "Communication", "Public Speaking",
            "Python (beginner)", "Research", "PowerPoint",
            "Multilingual", "Problem-Solving"
        ]
        self.interests = ["Technology", "Agriculture", "AI", "Football", "Quran"]
        self.tools = ["Figma", "VS Code", "GitHub", "Postman", "Notion"]
        self.mindset = "Fast learner with a passion for building impactful solutions."

    def introduce(self):
        print(f"Hi, I'm {self.name} 👋")
        print("I’m passionate about designing experiences that solve real-world problems.")
        print("Here are some roles I take on:")
        for role in self.roles:
            print(f" - {role}")
    
    def list_skills(self):
        print("\nSome of my core skills include:")
        for skill in self.skills:
            print(f"✅ {skill}")
    
    def my_mission(self):
        print("\n🚀 Mission: Use tech and design to make life easier for people around me.")
    
    def future_path(self):
        print("\n📈 I aim to blend healthcare, design, and technology to create something unique.")
        print("I’m not just learning — I’m building.")

# Instantiate and run
me = Abdurrahman()
me.introduce()
me.list_skills()
me.my_mission()
me.future_path()

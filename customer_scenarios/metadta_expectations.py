# Metadata: Customer Expectations vs Reality

print("=== METADATA: CUSTOMER CONFUSION ===")

# Customer expectation: "Metadata makes queries faster!"
print("❌ CUSTOMER MISCONCEPTION:")
print("   'Adding metadata filters will speed up my queries'")
print("   'More metadata = better performance'")

print("\n✅ REALITY:")
print("   Metadata filtering happens AFTER vector similarity")
print("   More complex filters = slower queries")
print("   Metadata is for precision, not speed")

# Example scenarios
scenarios = {
    "no_filter": {
        "query": "Find similar documents",
        "process": "Vector similarity only",
        "speed": "Fast",
        "results": "10,000 similar documents"
    },
    "simple_filter": {
        "query": "Find similar documents WHERE category='support'",
        "process": "Vector similarity + simple filter",
        "speed": "Medium", 
        "results": "500 similar support documents"
    },
    "complex_filter": {
        "query": "Find similar WHERE category='support' AND date > '2024-01-01' AND author IN ['alice','bob'] AND priority > 3",
        "process": "Vector similarity + complex multi-field filter",
        "speed": "Slow",
        "results": "5 highly targeted documents"
    }
}

print(f"\n📊 PERFORMANCE COMPARISON:")
for name, scenario in scenarios.items():
    print(f"\n{name.upper()}:")
    print(f"   Query: {scenario['query']}")
    print(f"   Speed: {scenario['speed']}")
    print(f"   Results: {scenario['results']}")

print(f"\n🎯 METADATA USE CASES:")
use_cases = [
    "Multi-tenancy: 'Only search customer A's data'",
    "Time filtering: 'Only documents from last month'", 
    "Access control: 'Only public documents'",
    "Content type: 'Only PDF files'",
    "Business logic: 'Only active products'",
    "A/B testing: 'Only experiment group B'"
]

for i, use_case in enumerate(use_cases, 1):
    print(f"{i}. {use_case}")

print(f"\n⚠️  CUSTOMER GOTCHAS:")
gotchas = [
    "Filtering 1M vectors to find 10 = still processes 1M vectors first",
    "Complex nested filters can timeout on large datasets", 
    "Metadata size counts toward the 40KB limit per vector",
    "Some filter operations much slower than others (ranges vs equals)",
    "Customers expect SQL-like performance, get vector-search reality"
]

for gotcha in gotchas:
    print(f"   • {gotcha}")

print(f"\n🚨 COMMON CSE ESCALATION:")
print(f"   Customer: 'Filtering makes my queries super slow!'")
print(f"   Reality: They're filtering AFTER finding 100K similar vectors")
print(f"   Solution: Reduce search scope first, then filter")
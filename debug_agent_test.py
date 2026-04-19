import pandas as pd
from agents.executor import agent_system
from utils.ollama import check_ollama_health

print('Ollama online:', check_ollama_health())
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 3, 4, 5, 6],
    'C': ['x', 'y', 'x', 'y', 'x']
})
agent_system.load_dataset(df, name='testcsv')
print('Dataset loaded:', agent_system.get_dataset().shape)
res = agent_system.run('Show me a summary of the data')
print('Run result keys:', list(res.keys()))
print('Final response:')
print(res.get('final_response'))
print('Agent output:')
print(res.get('agent_output'))
print('Selected agent:', res.get('selected_agent'))

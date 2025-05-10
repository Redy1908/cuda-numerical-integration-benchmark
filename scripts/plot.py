import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

file_path_gpu = "../results/gpu_results.csv"
df_gpu = pd.read_csv(file_path_gpu)
df_gpu['TotalThreads'] = df_gpu['ThreadsPerBlock'] * df_gpu['NumBlocks']
unique_total_threads_gpu = sorted(df_gpu['TotalThreads'].unique())

file_path_cpu = "../results/cpu_results.csv"
df_cpu = pd.read_csv(file_path_cpu)
df_cpu['TotalThreads'] = df_cpu['ThreadsPerBlock'] * df_cpu['NumBlocks']
unique_total_threads_cpu = sorted(df_cpu['TotalThreads'].unique())

file_path_gpu_property = "../results/gpu_property.csv"
df_gpu_prop = pd.read_csv(file_path_gpu_property)

props = df_gpu_prop.iloc[0] 
prop_list = [
    f"Nome: {props['GPUName']}",
    f"SMs: {props['SMs']}",
    f"Max Threads/Block: {props['MaxThreadsPerBlock']}",
    f"Max Threads/SM: {props['MaxThreadsPerSM']}",
    f"Warp Size: {props['WarpSize']}"
]
gpu_properties_text = "Proprietà GPU: " + " | ".join(prop_list)

y_formatter = mticker.ScalarFormatter(useOffset=False)
y_formatter.set_scientific(False)

# --- Grafico 1: Tempo d'esecuzione GPU vs Numero Totale di Thread ---
fig1, ax1 = plt.subplots(figsize=(12, 8)) 
for method in df_gpu['ImplementationMethod'].unique():
    method_data = df_gpu[df_gpu['ImplementationMethod'] == method]
    method_data = method_data.sort_values(by='TotalThreads')
    ax1.plot(method_data['TotalThreads'], method_data['ExecutionTimeSeconds'], marker='o', linestyle='-', label=method)

ax1.set_title('Tempo d\'esecuzione GPU al crescere del Numero Totale di Thread')
ax1.set_xlabel('Numero Totale di Thread (ThreadsPerBlock * NumBlocks)')
ax1.set_ylabel('Tempo d\'esecuzione (secondi)')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(y_formatter)
ax1.yaxis.set_minor_locator(mticker.NullLocator())
ax1.set_xticks(unique_total_threads_gpu)
ax1.set_xticklabels([str(t) for t in unique_total_threads_gpu]) 
ax1.legend() 
ax1.grid(True)

fig1.text(0.5, 0.02, gpu_properties_text, transform=fig1.transFigure,
          fontsize=11,
          verticalalignment='bottom', horizontalalignment='center',
          bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.09, 1, 0.95])
output_path_time_gpu = "../plots/gpu_execution_time_vs_total_threads.png"
plt.savefig(output_path_time_gpu)

# --- Grafico 2: Speed-up GPU vs Numero Totale di Thread (rispetto alla baseline GPU) ---
fig2, ax2 = plt.subplots(figsize=(12, 8)) 
for method in df_gpu['ImplementationMethod'].unique():
    method_data = df_gpu[df_gpu['ImplementationMethod'] == method].copy() 
    method_data = method_data.sort_values(by='TotalThreads')
    ax2.plot(method_data['TotalThreads'], method_data['Speedup'], marker='o', linestyle='-', label=method)

ax2.set_title('Speed-up GPU al crescere del Numero Totale di Thread')
ax2.set_xlabel('Numero Totale di Thread (ThreadsPerBlock * NumBlocks)')
ax2.set_ylabel('Speed-up')
ax2.set_xscale('log', base=2)
ax2.yaxis.set_major_formatter(y_formatter)
ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
ax2.set_xticks(unique_total_threads_gpu)
ax2.set_xticklabels([str(t) for t in unique_total_threads_gpu])
ax2.legend()
ax2.grid(True)

fig2.text(0.5, 0.02, gpu_properties_text, transform=fig2.transFigure,
          fontsize=11,
          verticalalignment='bottom', horizontalalignment='center',
          bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.09, 1, 0.95])
output_path_speedup_gpu = "../plots/gpu_speedup_vs_total_threads_from_csv.png"
plt.savefig(output_path_speedup_gpu)
# plt.show()

# --- Grafico 3: Efficienza GPU vs Numero Totale di Thread (rispetto alla baseline GPU) ---
fig3, ax3 = plt.subplots(figsize=(12, 8))
for method in df_gpu['ImplementationMethod'].unique():
    method_data = df_gpu[df_gpu['ImplementationMethod'] == method].copy()
    method_data = method_data.sort_values(by='TotalThreads')
    ax3.plot(method_data['TotalThreads'], method_data['Efficiency'], marker='o', linestyle='-', label=method)

ax3.set_title('Efficienza GPU al crescere del Numero Totale di Thread')
ax3.set_xlabel('Numero Totale di Thread (ThreadsPerBlock * NumBlocks)')
ax3.set_ylabel('Efficienza')
ax3.set_xscale('log', base=2)
ax3.yaxis.set_major_formatter(y_formatter)
ax3.yaxis.set_minor_formatter(mticker.NullFormatter())
ax3.set_ylim(bottom=0) 
ax3.set_xticks(unique_total_threads_gpu)
ax3.set_xticklabels([str(t) for t in unique_total_threads_gpu])
ax3.axhline(1, linestyle='--', color='gray', label='Efficienza Ideale (1.0)')
ax3.legend()
ax3.grid(True)

fig3.text(0.5, 0.02, gpu_properties_text, transform=fig3.transFigure,
          fontsize=11,
          verticalalignment='bottom', horizontalalignment='center',
          bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.09, 1, 0.95])
output_path_efficiency_gpu = "../plots/gpu_efficiency_vs_total_threads_from_csv.png"
plt.savefig(output_path_efficiency_gpu)

# --- Grafico 4: Speed-up GPU vs Numero Totale di Thread (rispetto alla CPU sequenziale) ---
fig4, ax4 = plt.subplots(figsize=(12, 8)) 
for method in df_cpu['ImplementationMethod'].unique():
    method_data_cpu = df_cpu[df_cpu['ImplementationMethod'] == method].copy() 
    method_data_cpu = method_data_cpu.sort_values(by='TotalThreads')
    ax4.plot(method_data_cpu['TotalThreads'], method_data_cpu['Speedup'], marker='o', linestyle='-', label=method)

ax4.set_title('Speed-up GPU vs CPU Sequenziale al crescere del Numero Totale di Thread')
ax4.set_xlabel('Numero Totale di Thread GPU (ThreadsPerBlock * NumBlocks)')
ax4.set_ylabel('Speed-up')
ax4.set_xscale('log', base=2)
ax4.yaxis.set_major_formatter(y_formatter)
ax4.yaxis.set_minor_locator(mticker.NullLocator())
ax4.set_xticks(unique_total_threads_cpu)
ax4.set_xticklabels([str(t) for t in unique_total_threads_cpu])
ax4.legend()
ax4.grid(True)

fig4.text(0.5, 0.02, gpu_properties_text, transform=fig4.transFigure,
          fontsize=11,
          verticalalignment='bottom', horizontalalignment='center',
          bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.09, 1, 0.95])
output_path_speedup_vs_cpu = "../plots/gpu_speedup_vs_cpu_sequential.png"
plt.savefig(output_path_speedup_vs_cpu)
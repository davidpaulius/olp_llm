import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid Type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def compute_mean_err_bar(data, confidence=None):
	# Sample data
	data = np.array(data)

	# Sample mean and standard error
	mean = np.mean(data)
	std_error = stats.sem(data)  # standard error of the mean

	if confidence:  # 95% confidence interval
		# Compute the 95% confidence interval; use t-distribution since sample size < 30
		n = len(data)
		t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)  # t-critical value for 95% CI

		margin_of_error = t_critical * std_error
		# confidence_interval = (mean - margin_of_error, mean + margin_of_error)
		err_bar = margin_of_error
	else:
		err_bar = std_error
	return mean, err_bar


def make_plot():
	spelling = {
		"OLP": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [100, 100, 90, 80, 30],
		},
		"LLM-Planner": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [20, 20, 70, 30, 40],
		},
		"LLM+P": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [30, 60, 60, 40, 40],
		},
		"DELTA": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [80, 100, 90, 60, 60],
		},
	}

	towers = {
		"OLP": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [90, 90, 90, 100, 60],
		},
		"LLM-Planner": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [100, 60, 40, 10, 10],
		},
		"LLM+P": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [10, 40, 70, 40, 20],
		},
		"DELTA": {
			"x": ["3", "4", "5", "6", "7"],
			"y": [100, 100, 90, 90, 50],
		},
	}

	organize = {
		"OLP": {
			"x": ["6", "7", "8", "9", "10", "11", "12"],
			"y": [90, 100, 70, 100, 90, 60, 60],
		},
		"LLM-Planner": {
			"x": ["6", "7", "8", "9", "10", "11", "12"],
			"y": [90, 40, 70, 20, 20, 10, 0],
		},
		"LLM+P": {
			"x": ["6", "7", "8", "9", "10", "11", "12"],
			"y": [40, 40, 50, 60, 60, 10, 0],
		},
		"DELTA": {
			"x": ["6", "7", "8", "9", "10", "11", "12"],
			"y": [90, 40, 70, 90, 70, 60, 10],
		},
	}


	# Set plot parameters
	fig, ax = plt.subplots(figsize=(12,4))
	width = 0.08 # width of bar

	patterns = [" ", "/////" , "\\\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

	tasks = ["towers", "spelling", "organize"]
	methods = ["OLP", "LLM-Planner", "LLM+P", "DELTA"]
	for data in tasks:
		for z in methods:
			pattern = tasks.index(data)
			colour = methods.index(z)
			x_data = eval(data)[z]["x"]
			for x in range(len(x_data)):
				x_data[x] = int(x_data[x]) + (width * colour) + (width * pattern * len(methods))
			ax.bar(x_data, eval(data)[z]["y"], width, color=sns.color_palette('colorblind')[colour], label=f"{z} - {data}", hatch=patterns[pattern])

	ax.set_ylabel('Plan Completion (%)', fontsize=17)
	ax.set_ylim(0,105)
	ax.set_xticks(np.arange(3, 13) + (width * 4) + width/2)
	ax.set_xticklabels(np.arange(3, 13))
	ax.set_xlim(2.75,13.25)
	ax.set_xlabel('Number of Blocks in Scene', fontsize=17)
	ax.set_title('Percentage of Plans Completely Executed Across Tasks and Approaches', fontsize=18, fontweight='semibold')
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	# see here about legend: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

	# Shrink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
					box.width, box.height * 0.9])

	# Put a legend below current axis
	# ax.legend(loc='center right', bbox_to_anchor=(0.5, -0.4),
	# 		fancybox=True, shadow=True, ncol=5)

	# ax.legend()
	fig.tight_layout()

	plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
	# plt.xticks(rotation=45,ha='right')
	fig.tight_layout()
	plt.show()

if __name__ == "__main__":
	make_plot()

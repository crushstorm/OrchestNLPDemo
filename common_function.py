import orchest
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def plot_sentiment(df,color,title) :
	color = ['#008972']
	ax = df.groupby("variation").pos.mean().plot.bar(color = color, figsize = (9, 6))

	plt.title(title, fontsize = 20, weight='bold')
	plt.xticks(rotation='90', fontsize=14, weight='bold')
	ax.xaxis.label.set_visible(False)

	plt.ylabel('Rating', fontsize=16, weight='bold')
	ax.set_ylim([0,0.5])
	plt.yticks(fontsize=14)


	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(False)


	fig = plt.gcf()
	plt.show()
	plt.draw()
    
# DEFINE FUNCTION TO CALCULATE SENTIMENT SCORE 
def sentimentScore(sentences):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        #print(str(vs))
        results.append(vs) 
    return results
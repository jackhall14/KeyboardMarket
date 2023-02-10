# 
# Original Author: Jack Hall
# Made Public: 26th June 2020
# 
# To do (in order of priority):
# 0.5) Finalise date plot for xticks
# 1) Add more plots - correlation between variables, histograms of price? 
# 2) Improve text parsing in methods PriceCheck and ParsePostBody
# 3) Improve price estimation with CurrencyConverter module:
# https://pypi.org/project/CurrencyConverter/
# 4) Improve options for fitting

import praw, argparse, re, sys, datetime, os, logging, json
from parse import *
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
import seaborn as sns
logging.basicConfig(level=logging.INFO)

def main():
    args = get_args()
    if args.Debug: logging.getLogger().setLevel(logging.DEBUG)

    subreddit = GetSubreddit()
    KeebList = GetListOfKeebs(args)
    DataDict = GetData(args, subreddit, KeebList)
    DF = CreateDF(DataDict)
    for Keeb in KeebList:
	    if args.PlotData: MakePlots(args, DF, Keeb)
	    if args.SaveData: SaveData(args, DF, Keeb)
    logging.info("All finished.")

def MakePlots(Args, DataFrame, Keeb):
	logging.info("Prepping DF for plots ...")
	DataFrame = DataFrame.loc[DataFrame['Sale Item'] == Keeb.upper()]
	DataFrame = DataFrame.sort_values(by='Post Date',ascending=True)
	logging.info(DataFrame)
	logging.info(DataFrame.describe())
	pd.set_option('display.max_colwidth', None)
	logging.info(DataFrame["URL"])

	DatePlot(DataFrame, Keeb)
	# ItemPlotMPL(DataFrame, List_of_Keebs, OutputDir)

def SaveData(Args, DataFrame, Keeb):
	cwd = os.getcwd() + "/"
	OutputDir = cwd + "csv_data/"
	if not os.path.exists(OutputDir): os.makedirs(OutputDir)

	# NewKeebName = ("_").join(Keeb.split(" "))
	NewKeebName = Keeb
	OutputDataName = NewKeebName + ".csv"
	DataFrame.to_csv(OutputDir + OutputDataName, index=False)
	logging.info("Outputed file to:\n" + OutputDir + OutputDataName)

def GetData(Args, subreddit, List_of_Keebs):
	List_Of_Output_Dicts = []
	for Keeb in List_of_Keebs:
		search_subreddit = subreddit.search("[H] " + Keeb + " [W] PayPal", limit=Args.DataLimit)
		Keeb_Renamed = Keeb.upper()
		for submission in search_subreddit:
			# Loop over posts in the subreddit
			if CheckGoodSubmission(submission, Keeb_Renamed):
				if not submission.stickied:
					# Stickied is weekly stuff
					PrintSubmissionInfo(submission)

					# Need to parse title to find whats been sold
					Location, HaveObjects, WantObjects = ParseTitle(submission)
					# Need to parse post body to find prices
					ObjectDict = ParsePostBody(submission,HaveObjects,WantObjects,Location, Keeb_Renamed)
					
					PostDate = PrintOutputDict(ObjectDict, submission, Location)
					if Args.DebugKeeb: input()

					# Now we have most of the data for each sale, append the last parts
					for key, value in ObjectDict.items():
						List_Of_Output_Dicts.append(
							{
								'Post Location': Location,
								'Post Date': PostDate,
								'User': submission.fullname,
								'URL': submission.url,
								'Sale Item':  key,
								'Asking Price': value[0],
								'Sold': value[1]
							}
						)
	return List_Of_Output_Dicts

def CreateDF(Dict):
	# Create a dataframe from the output dictionary
	DataFrame = pd.DataFrame(Dict)
	DataFrame["Asking Price"] = pd.to_numeric(DataFrame["Asking Price"])
	# Get names of indexes for which column Age has value 30
	indexNames = DataFrame[ DataFrame['Asking Price'] == 1 ].index
	# Delete these row indexes from dataFrame
	DataFrame.drop(indexNames , inplace=True)
	DataFrame = DataFrame.dropna()
	# Output:
	logging.info(80*"-")
	logging.info("Dataframe summary:")
	logging.info(DataFrame)
	logging.info(DataFrame.describe())
	return DataFrame

def GetListOfKeebs(Args):
	# If you want to hard-code a list of many, do it here:
	Custom_List_Of_Keebs = []
	if Custom_List_Of_Keebs: List_of_Keebs = Custom_List_Of_Keebs
	else: List_of_Keebs = Args.Keeb
	return List_of_Keebs

def GetSubreddit():
	F = open("login.json")
	JSONF = json.load(F)
	reddit = praw.Reddit(user_agent=JSONF["user_agent"], client_id=JSONF["client_id"],
	                     client_secret=JSONF["client_secret"], username=JSONF["username"],
	                     password=JSONF["password"])
	subreddit = reddit.subreddit("mechmarket") 
	F.close()
	return subreddit

def PrintSubmissionInfo(Submission):
	logging.info(80*"-")
	logging.info("\t\t\t\t SUBMISSION:")
	logging.info("Title:\t" + str(Submission.title))
	logging.debug("Ups:\t" + str(Submission.ups) + "\t\tDowns:\t" + str(Submission.downs))
	logging.info("URL:\t" + str(Submission.url))
	logging.info(80*"-")

def PrintOutputDict(Dictionary, Submission, Location):
	logging.info(80*"-")
	logging.info("OUTPUT DATA:")
	SubDate = GetDate(Submission)
	logging.info("Post Location:\t\t" + str(Location))
	logging.info("Post Date:\t\t" + str(SubDate))
	logging.info("User:\t\t\t" + str(Submission.fullname))
	logging.info("KEEBS:")
	for key, value in Dictionary.items():
		logging.info("Keyboard:\t\t" + str(key))
		logging.info("Asking Price:\t\t" + str(value[0]))
		logging.info("Sold:\t\t\t" + str(value[1]))
	logging.info(80*"-")
	return SubDate

def DatePlot(DF, Keeb):
	logging.info("Making plot of asking price vs date posted for keeb:\t"+Keeb)

	# Begin Plot:
	x_var_name = "Post Date"
	y_var_name = "Asking Price"
	x_var = DF[x_var_name]
	y_var = DF[y_var_name]

	# Axis and Title labels:
	# KeebName = Keeb.upper()
	# figureTitle = KeebName
	figureTitle = Keeb
	figurexlabel = x_var_name
	figureylabel = y_var_name + " (\$)"
	Prop_array = [figureTitle, figurexlabel, figureylabel]

	# Plot styles
	plt.rcParams["figure.figsize"] = (12,6)
	plt.style.use("ggplot")

	# Code to plot:
	# g = sns.scatterplot(data=DF, x=x_var, y=y_var,hue="Sold")
	# g = sns.relplot(data=DF, x=x_var, y=y_var,hue="Sold")
	g = sns.stripplot(x=x_var,y=y_var,data=DF,hue="Sold", jitter=False)
	
	g.figure.suptitle("Sales of "+Prop_array[0],fontsize=16)
	g.figure.set_size_inches(12,6)
	plt.subplots_adjust(bottom=0.26)

	# Axes
	plt.xlabel(Prop_array[1],fontsize=14)
	plt.ylabel(Prop_array[2],fontsize=14)
	ax = plt.gca()
	ax.set_ylim([0, y_var.max() * 1.05])
	plt.gcf().autofmt_xdate()
	g.set_xticklabels(g.get_xticklabels(), rotation=90)
	# date_format = mpl_dates.DateFormatter('%b, %d %Y')
	# ax.xaxis.set_major_formatter(date_format)

	# Final matter:
	plt.show()
	logging.info(plt.figure())
	logging.info(plt.axes())

	NewKeebName = Keeb
	# NewKeebName = ("_").join(KeebName.split(" "))
	cwd = os.getcwd() + "/"
	OutputDir = cwd + "Plots/"+NewKeebName+"/"
	if not os.path.exists(OutputDir): os.makedirs(OutputDir)
	OutputPlotName = NewKeebName + "_sales_timeseries" + ".png"
	g.figure.savefig(OutputDir+OutputPlotName)
	logging.info("Saved to:\t" +OutputDir+OutputPlotName)

# def ItemPlotMPL(DF, List_of_Keebs, OutputDir):
# 	# Actual Plot:
# 	x_var_name = "Post Date"
# 	y_var_name = "Asking Price"
# 	x_var = DF[x_var_name]
# 	y_var = DF[y_var_name]

# 	# Axis and Title labels:
# 	KeebName = List_of_Keebs[0].upper()
# 	figureTitle = KeebName
# 	figurexlabel = x_var_name
# 	figureylabel = y_var_name + " (\$)"
# 	Prop_array = [figureTitle, figurexlabel, figureylabel]

# 	# Plot settings
# 	plt.rcParams["figure.figsize"] = (12,6)
# 	plt.style.use("ggplot")

# 	# Code to plot:
# 	colours = y_var * (100. / y_var.max())
# 	plt.scatter(x_var,y_var,c=colours,cmap="RdYlBu",alpha=0.9)

# 	plt.subplots_adjust(bottom=0.26)

# 	# Axes
# 	plt.xlabel(Prop_array[1],fontsize=14)
# 	plt.ylabel(Prop_array[2],fontsize=14)
# 	ax = plt.gca()
# 	ax.set_ylim([0, y_var.max() * 1.05])
# 	plt.gcf().autofmt_xdate()
# 	date_format = mpl_dates.DateFormatter('%b, %d %Y')
# 	ax.xaxis.set_major_formatter(date_format)

# 	# Other
# 	plt.title("Sales of "+Prop_array[0])

# 	# Final matter:
# 	plt.show()
# 	logging.info(plt.figure())
# 	logging.info(plt.axes())

# 	# Output
# 	NewKeebName = ("_").join(KeebName.split(" "))
# 	OutputPlotName = NewKeebName + ".png"
# 	# g.figure.savefig(OutputDir+OutputPlotName)
# 	logging.info("Saved to:\t" +OutputDir+OutputPlotName)


def CheckGoodSubmission(Submission, Keyboard):
	Title = Submission.title.upper()
	# Get lots of index locations
	HaveLocationIndex = Title.find("[H]")
	WantLocationIndex = Title.find("[W]")
	FirstPayPalLocationIndex = Title.find("PAYPAL")
	SecPayPalLocationIndex = Title.rfind("PAYPAL")
	if "TGR" in Keyboard: tmp_keeb_name = Keyboard.split("TGR ")[1]
	else: tmp_keeb_name = Keyboard
	KeebLocationIndex = Title.find(tmp_keeb_name.upper())
	TradesLocationIndex = Title.find("TRADE")
	if KeebLocationIndex > WantLocationIndex:
		# [W] Keeb
		return False
	if KeebLocationIndex > TradesLocationIndex and TradesLocationIndex != -1:
		# Trades, Keeb
		return False
	if TradesLocationIndex > WantLocationIndex:
		# [W] Trades
		return False
	else:
		# These are the good ones
		return True

def GetDate(submission):
	time = submission.created
	WholeDTObj = datetime.datetime.fromtimestamp(time)
	date = WholeDTObj.date()
	# Other options:
	# logging.info(blah.timestamp())
	# logging.info(blah.year)
	# logging.info(blah.month)
	# logging.info(blah.day)
	return date

def ParseTitle(Submission):
	# Need to update this to a regular expression
	Title = Submission.title.upper()

	pattern = re.compile(r'\[(.*)\].*(\[H\])(.*)(\[W\])(.*)')
	match = pattern.search(Title)

	Location = match.group(1)
	HaveObjects = match.group(3).strip()
	WantObjects = match.group(5).strip()

	if "(" in HaveObjects or "+" in HaveObjects:
		# Found a title with brackets in (THESE ARE PROBLEMATIC)
		HaveObjects = HaveObjects.replace("(","\(")
		HaveObjects = HaveObjects.replace(")","\)")
		HaveObjects = HaveObjects.replace("+","")
	return Location, HaveObjects, WantObjects

def PriceCheck(Text, Currency, Keyboard = "None", SoldCheck = False):
	if Keyboard == "None": SearchStr = '('+Currency+'(\d+)|(\d+)'+Currency+')'
	elif Keyboard != "None" and SoldCheck == False: SearchStr = Keyboard+'.*'+Currency+'(\d{1,9})'
	else: SearchStr = Keyboard+'.*'+Currency+'(\d{1,9}).*SOLD'
	pattern = re.compile(r''+SearchStr)
	match = pattern.search(Text)
	return match

def ParsePostBody(Submission, HaveObjects, WantObjects, Location, Keyboard):
	logging.debug("PARSING POST BODY LOG:\n")
	PostBody = Submission.selftext.upper()
	ObjectDict = {}
	if len(HaveObjects.split(",")) == 1:
		# Only one object so need to find the price
		ListInfo = []
		logging.debug("Searching for:\t\t" + HaveObjects)
		match = PriceCheck(PostBody, "\$")
		if match:
			# Currently takes the first
			AskingPrice = match.group(2)
			logging.debug("Found asking price:\t" + str(AskingPrice))
			ListInfo.append(AskingPrice)
		else:
			if PriceCheck(PostBody,"\€"):
				match = PriceCheck(PostBody,"\€")
				AskingPrice = match.group(2)
				logging.debug("Found asking price:\t" + str(AskingPrice))
				ListInfo.append(AskingPrice)
			if PriceCheck(PostBody,"\£"):
				match = PriceCheck(PostBody,"\£")
				AskingPrice = match.group(2)
				logging.debug("Found asking price:\t" + str(AskingPrice))
				ListInfo.append(AskingPrice)
			else:
				logging.warning("Couldnt find asking price so filling it with the number 1.")
				ListInfo.append(1)
		# Determine the Sold Price
		if Submission.link_flair_text:
			if "Sold" in Submission.link_flair_text:
				logging.debug("Submission was sold, assuming it was asking price")
				Sold = True
				ListInfo.append(Sold)
			else:
				# Search for Sold in body
				logging.debug("Flair has not been updated, checking in the body")
				Sold = search("SOLD",PostBody)
				if Sold is None:
					logging.debug("Not been confirmed in body either, checking comments")
					Sold = False
					ListInfo.append(Sold)
				else:
					Sold = True
					ListInfo.append(Sold)
		else:
			Sold = False
			ListInfo.append(Sold)
		ObjectDict.update({Keyboard:ListInfo})
		return ObjectDict
	else:
		# For posts with multiple items for sale
		for Object in HaveObjects.split(","):
			Object = Object.strip()
			ListInfo = []
			logging.info("Searching for:\t\t" + Object)
			if PriceCheck(PostBody, "\$", Object):
				match = PriceCheck(PostBody, "\$", Object)
				AskingPrice = match.group(1)
				ListInfo.append(AskingPrice)
				logging.info("Found asking price:\t" + str(AskingPrice))
				# FullString = match.group(0)
				# Now need to check if it sold for that
				if PriceCheck(PostBody, "\$", Object, True):
					Sold = True
					ListInfo.append(Sold)
				else:
					logging.info("Couldn't confirm " + Object + " was sold.")
					Sold = False
					ListInfo.append(Sold)
				ObjectDict.update({Object:ListInfo})
			else:
				if Keyboard in Object:
					if PriceCheck(PostBody, "\$", Keyboard):
						match = PriceCheck(PostBody, "\$", Keyboard)
						AskingPrice = match.group(1)
						ListInfo.append(AskingPrice)
						logging.info("Found asking price:\t" + str(AskingPrice))
						if PriceCheck(PostBody, "\$", Object, True):
							Sold = True
							ListInfo.append(Sold)
						else:
							logging.info("Couldn't confirm " + Object + " was sold.")
							Sold = False
							ListInfo.append(Sold)
						ObjectDict.update({Keyboard:ListInfo})
				else:
					logging.info("Couldn't find asking price")
					# Need to declare values here
		return ObjectDict

def get_args():
	args = argparse.ArgumentParser()
	args.add_argument('--Keeb', type=str, nargs='+', default=["polaris"], help='Name of keebs you would like to search for - seperate by space e.g: --Keeb polaris aella.')
	args.add_argument('--SaveData', action='store_true', help='When youre happy to export your dataframe, turn this on')
	args.add_argument('--PlotData', action='store_true', help='Get a sales version of the final DF, for single keebs only.')
	args.add_argument('--DebugKeeb', action='store_true', help='When gathering data, pause after each keeb to check the log.')
	args.add_argument('--DataLimit', type=int, default=40000, help='Restrict the number of listings it will perform per keyboard.')
	args.add_argument('--Debug', action='store_true', help="Changes the log level to debug")
	return args.parse_args()

if __name__ == '__main__': main()

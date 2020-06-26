# 
# Original Author: Jack Hall
# Made Public: 26th June 2020
# 

import praw
import argparse
import re
import sys
import datetime
from parse import *
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--Keeb', type=str, nargs='+', help='Name of singular keeb you would like to search for.')
parser.add_argument('--TraceKeeb', action='store_true', help='When gathering data, pause after each keeb to check the log.')
parser.add_argument('--AllHighEndKeebs', action='store_true', help='Run over a predefined list of high-end keyboards to build a dataset.')
parser.add_argument('--DataLimit', type=int, default=40000, help='Restrict the number of listings it will perform per keyboard.')
parser.add_argument('--SaveDF', action='store_true', help='When youre happy to export your dataframe, turn this on')
parser.add_argument('--SaleDF', action='store_true', help='Get a sales version of the final DF, for single keebs only.')
args = parser.parse_args()

def PrintSubmissionInfo(Submission):
	print(80*"-")
	print("\t\t\t\t SUBMISSION:")
	print("Title:\t" + str(submission.title))
	print("Ups:\t" + str(submission.ups) + "\t\tDowns:\t" + str(submission.downs))
	print("URL:\t" + str(submission.url))
	print(80*"-")

def PrintOutputDict(Dictionary, Submission, Location):
	print(80*"-")
	print("\t\t\t\t Output Data:")
	print("GENERAL INFO:")
	SubDate = GetDate(submission)
	print("Post Location:\t\t" + str(Location))
	print("Post Date:\t\t" + str(SubDate))
	print("User:\t\t\t" + str(submission.fullname))
	print("KEEBS:")
	for key, value in Dictionary.items():
		print("Keyboard:\t\t" + str(key))
		print("Asking Price:\t\t" + str(value[0]))
		print("Sold:\t\t\t" + str(value[1]))
	print(80*"-")
	return SubDate

def ItemPlot(DF):
	# Actual Plot:
	x_var_name = "Post Date"
	y_var_name = "Asking Price"
	x_var = DF[x_var_name]
	y_var = DF[y_var_name]

	# Axis and Title labels:
	KeebName = List_of_Keebs[0].upper()
	figureTitle = KeebName
	figurexlabel = x_var_name
	figureylabel = y_var_name + " (\$)"
	Prop_array = [figureTitle, figurexlabel, figureylabel]

	plt.rcParams["figure.figsize"] = (12,6)

	# Code to plot:
	g = sns.stripplot(x=x_var,y=y_var,data=DF)
	g.figure.suptitle(Prop_array[0],fontsize=16)
	
	g.figure.set_size_inches(12,6)
	plt.subplots_adjust(bottom=0.26)

	g.set_xticklabels(g.get_xticklabels(), rotation=90)
	plt.xlabel(Prop_array[1],fontsize=14)
	plt.ylabel(Prop_array[2],fontsize=14)

	# Final matter:
	plt.show(g)
	print(plt.figure())
	print(plt.axes())

	OutputDir = cwd + "Plots/"
	if not os.path.exists(OutputDir):
		# Create output directory  if it doesn't exist
		os.makedirs(OutputDir)

	NewKeebName = ("_").join(KeebName.split(" "))
	OutputPlotName = NewKeebName + ".png"
	g.figure.savefig(OutputDir+OutputPlotName)
	print("Saved to:\t" +OutputDir+OutputPlotName)

def CheckGoodSubmission(Submission, Keyboard):
	Title = Submission.title.upper()
	# Get lots of index locations
	HaveLocationIndex = Title.find("[H]")
	WantLocationIndex = Title.find("[W]")
	FirstPayPalLocationIndex = Title.find("PAYPAL")
	SecPayPalLocationIndex = Title.rfind("PAYPAL")
	if "TGR" in Keyboard:
		tmp_keeb_name = Keyboard.split("TGR ")[1]
	else:
		tmp_keeb_name = Keyboard
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
	# print(blah.timestamp())
	# print(blah.year)
	# print(blah.month)
	# print(blah.day)
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
	if Keyboard == "None":
		SearchStr = '('+Currency+'(\d+)|(\d+)'+Currency+')'
	elif Keyboard != "None" and SoldCheck == False:
		SearchStr = Keyboard+'.*'+Currency+'(\d{1,9})'
	else:
		SearchStr = Keyboard+'.*'+Currency+'(\d{1,9}).*SOLD'
	pattern = re.compile(r''+SearchStr)
	match = pattern.search(Text)
	return match

def ParsePostBody(Submission, HaveObjects, WantObjects, Location, Keyboard):
	print("PARSING POST BODY LOG:\n")
	PostBody = submission.selftext.upper()
	ObjectDict = {}
	if len(HaveObjects.split(",")) == 1:
		# Only one object so need to find the price
		ListInfo = []
		print("Searching for:\t\t" + HaveObjects)
		match = PriceCheck(PostBody, "\$")
		if match:
			# Currently takes the first
			AskingPrice = match.group(2)
			print("Found asking price:\t" + str(AskingPrice))
			ListInfo.append(AskingPrice)
		else:
			if PriceCheck(PostBody,"\€"):
				match = PriceCheck(PostBody,"\€")
				AskingPrice = match.group(2)
				print("Found asking price:\t" + str(AskingPrice))
				ListInfo.append(AskingPrice)
			if PriceCheck(PostBody,"\£"):
				match = PriceCheck(PostBody,"\£")
				AskingPrice = match.group(2)
				print("Found asking price:\t" + str(AskingPrice))
				ListInfo.append(AskingPrice)
			else:
				print("Couldnt find asking price so filling it with the number 1.")
				ListInfo.append(1)
		# Determine the Sold Price
		if submission.link_flair_text:
			if "Sold" in submission.link_flair_text:
				print("Submission was sold, assuming it was asking price")
				Sold = True
				ListInfo.append(Sold)
			else:
				# Search for Sold in body
				print("Flair has not been updated, checking in the body")
				Sold = search("SOLD",PostBody)
				if Sold is None:
					print("Not been confirmed in body either, checking comments")
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
			print("Searching for:\t\t" + Object)
			if PriceCheck(PostBody, "\$", Object):
				match = PriceCheck(PostBody, "\$", Object)
				AskingPrice = match.group(1)
				ListInfo.append(AskingPrice)
				print("Found asking price:\t" + str(AskingPrice))
				# FullString = match.group(0)
				# Now need to check if it sold for that
				if PriceCheck(PostBody, "\$", Object, True):
					Sold = True
					ListInfo.append(Sold)
				else:
					print("Couldn't confirm " + Object + " was sold.")
					Sold = False
					ListInfo.append(Sold)
				ObjectDict.update({Object:ListInfo})
			else:
				if Keyboard in Object:
					if PriceCheck(PostBody, "\$", Keyboard):
						match = PriceCheck(PostBody, "\$", Keyboard)
						AskingPrice = match.group(1)
						ListInfo.append(AskingPrice)
						print("Found asking price:\t" + str(AskingPrice))
						if PriceCheck(PostBody, "\$", Object, True):
							Sold = True
							ListInfo.append(Sold)
						else:
							print("Couldn't confirm " + Object + " was sold.")
							Sold = False
							ListInfo.append(Sold)
						ObjectDict.update({Keyboard:ListInfo})
				else:
					print("Couldn't find asking price")
					# Need to declare values here
		return ObjectDict

# Basic set up
reddit = praw.Reddit(user_agent="praw",
                     client_id="<Your Client Id>", client_secret="<Your Client Secret>",
                     username="<Your Username>", password="<Your Password>")
subreddit = reddit.subreddit("mechmarket")
cwd = os.getcwd()
cwd = cwd + "/"

# Create Groups of Keebs
HighEnd60P = ["M60-A", "TGR Unikorn"]
HighEnd65P = ["TGR 910", "Think 6.5", "Kyuu", "E6.5", "bauer", "Fjell"]
HighEnd70P = ["Singa V3", "Duck Octogan v3", "Satisfaction 75"]
HighEndTKLs = ["U80-A","Keycult No.1", "Keycult No.2", "Duck Orion V2", "TGR Jane V2", "TGR Jane CE", "Lynn Whale"]
HighEndAlices = ["TGR Alice"]
AllHighEndKeebs =  HighEnd60P + HighEnd65P + HighEnd70P + HighEndAlices + HighEndTKLs

# Gather a list of keebs you'd like data for
List_of_Keebs = []
if args.Keeb:
	List_of_Keebs.append((" ").join(args.Keeb))
elif args.AllHighEndKeebs:
	List_of_Keebs = AllHighEndKeebs

List_Of_Output_Dicts = []
for Keeb in List_of_Keebs:
	search_subreddit = subreddit.search("[H] " + Keeb + " [W] PayPal", limit=args.DataLimit)
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

				if args.TraceKeeb:
					input()

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

# Create a dataframe from the output dictionary
DataFrame = pd.DataFrame(List_Of_Output_Dicts)
DataFrame["Asking Price"] = pd.to_numeric(DataFrame["Asking Price"])
# Get names of indexes for which column Age has value 30
indexNames = DataFrame[ DataFrame['Asking Price'] == 1 ].index
# Delete these row indexes from dataFrame
DataFrame.drop(indexNames , inplace=True)
DataFrame = DataFrame.dropna()
if args.SaleDF:
	DataFrame = DataFrame.loc[DataFrame['Sale Item'] == List_of_Keebs[0].upper()]
	DataFrame = DataFrame.sort_values(by='Post Date',ascending=True)
	print(DataFrame)
	print(DataFrame.describe())
	pd.set_option('display.max_colwidth', None)
	print(DataFrame["URL"])

	ItemPlot(DataFrame)
else:
	print(DataFrame)
	print(DataFrame.describe())

if args.SaveDF:
	# First create a directory for it if it doesnt exist already
	OutputDir = cwd + "output_data/"
	if not os.path.exists(OutputDir):
		# Create output directory  if it doesn't exist
		os.makedirs(OutputDir)

	if args.Keeb:
		NewKeebName = ("_").join(Keeb.split(" "))
		OutputDataName = NewKeebName + ".csv"
	else:	
		OutputDataName = "AllHighEndKeebs.csv"
	DataFrame.to_csv(OutputDir + OutputDataName, index=False)
	print("Outputed file to:\n" + OutputDir + OutputDataName)
print("\nFinished.")

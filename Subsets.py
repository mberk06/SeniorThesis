"""
	Author: Michael Berjk
	Date: Spring 2020
	Description:
		- develop lists for a variety of analysis susbets
"""

############################################
############################################
# Imports and Global setup 
############################################
############################################
import sys

# create globals module
SUBSETS = sys.modules[__name__]

############################################
############################################
# All Organisms 
############################################
############################################
SUBSETS.ALL_ORGANISMS = ['ARABIAN BUTTERFLYFISH','ASPERGILLOSIS','BANDED CORAL SHRIMP','BARRACUDA',
				 'BARRAMUNDI COD','BLACK BAND','BLACK BAND','BLACK SPOTTED GRUNT',
				 'BLACK URCHIN','BLEACHING (% OF COLONY)','BLEACHING (% OF POPULATION)',
				 'BLUE SEA STAR','BLUELINE SNAPPER','BROOMTAIL WRASSE',
				 'BUMPHEAD PARROT','BUTTERFLYFISH','COTS',
				 'COWRIES','DARK BUTTERFLYFISH','DARK BUTTERFLYFISH','DIADEMA',
				 'EDIBLE SEA CUCUMBER','FLAMINGO TONGUE','GIANT CLAM 10-20 CM',
				 'GIANT CLAM 20-30 CM','GIANT CLAM 30-40 CM','GIANT CLAM 40-50 CM',
				 'GIANT CLAM <10 CM','GIANT CLAM >50 CM','GIANT CLAM TOTAL',
				 'GIANT HAWKFISH','GOATFISH','GORGONIAN','GREY GRUNT','GROUPER 30-40 CM',
				 'GROUPER 40-50 CM','GROUPER 50-60 CM','GROUPER >60 CM','GROUPER TOTAL',
				 'GRUNTS','HAEMULIDAE','HELMET CONCH','HUMPHEAD WRASSE',
				 'JACKS','KING ANGELFISH','LIONFISH','LOBSTER','LONGFIN BANNERFISH',
				 'MANTAS','MEXICAN HOGFISH','MORAY EEL','NASSAU GROUPER 30-40 CM',
				 'NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
				 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','NASSAU GROUPER 30-40 CM',
				 'NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
				 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','ORANGE SPINE UNICORNFISH',
				 'ORANGE SPOTTED GROUPER 30-40 CM','ORANGE SPOTTED GROUPER 40-50 CM',
				 'ORANGE SPOTTED GROUPER 50-60 CM','ORANGE SPOTTED GROUPER >60 CM',
				 'ORANGE SPOTTED GROUPER TOTAL','PARROTFISH',
				 'PEACOCK GROUPER 30-40 CM','PEACOCK GROUPER 40-50 CM',
				 'PEACOCK GROUPER 50-60 CM','PEACOCK GROUPER >60 CM',
				 'PEACOCK GROUPER TOTAL','PENCIL URCHIN','QUEEN CONCH','SEA FAN','SHARKS',
				 'SHORT SPINE URCHIN','SLATE PENCIL URCHIN','SNAPPER',
				 'SPIDER CRAB','SPOTTED GRUNT','TRIPNEUSTES','TRITON','TROCHUS','TURTLES',
				 'WHITE BAND','WHITE PLAGUE','WHITE BAND','YELLOW GOATFISH','YELLOW TANG',
				 'YELLOWBAR ANGELFISH','YELLOWTAIL TANG']

############################################
############################################
# Trophic Level Subsets 
############################################
############################################
SUBSETS.PRODUCERS = ['GORGONIAN','SEA FAN']

SUBSETS.PRIMARY_CONSUMERS = ['ARABIAN BUTTERFLYFISH','BANDED CORAL SHRIMP','BLACK URCHIN','BROOMTAIL WRASSE',
					 'BUMPHEAD PARROT','BUTTERFLYFISH','COTS','COWRIES','DARK BUTTERFLYFISH','DIADEMA',
					 'EDIBLE SEA CUCUMBER','FLAMINGO TONGUE','GIANT CLAM 10-20 CM',
                     'GIANT CLAM 20-30 CM','GIANT CLAM 30-40 CM','GIANT CLAM 40-50 CM',
                     'GIANT CLAM <10 CM','GIANT CLAM >50 CM','GIANT CLAM TOTAL','LOBSTER','ORANGE SPINE UNICORNFISH',
					 'MANTA','PARROTFISH','PENCIL URCHIN','SHORT SPINE URCHIN','SPIDER CRAB','TRIPNEUSTES',
					 'TURTLES','YELLOW TANG','YELLOWTAIL TANG']

SUBSETS.SECONDARY_CONSUMERS = ['ARABIAN BUTTERFLYFISH','BANDED CORAL SHRIMP','BLACK SPOTTED GRUNT','BLUELINE SNAPPER',
					   'BROOMTAIL WRASSE','BUTTERFLYFISH','DARK BUTTERFLYFISH','EDIBLE SEA CUCUMBER','GOATFISH',
					   'GREY GRUNT','GRUNTS','HELMET CONCH','HUMPHEAD WRASSE','KING ANGELFISH',
					   'LONGFIN BANNERFISH','KING ANGELFISH','MEXICAN HOGFISH','QUEEN CONCH','SPIDER CRAB',
					   'SPOTTED GRUNT','TRITON','TURTLES','YELLOW GOATFISH','YELLOWBAR ANGELFISH','BLUE SEA STAR']

SUBSETS.TERCIARY_CONSUMERS = ['BARRAMUNDI COD','GIANT HAWKFISH','GROUPER 30-40 CM','JACKS','LIONFISH',
					  'NASSAU GROUPER 30-40 CM','ORANGE SPOTTED GROUPER 30-40 CM','PEACOCK GROUPER 30-40 CM']

SUBSETS.QUARTERNARY_CONSUMERS = ['BARRACUDA','GROUPER 40-50 CM','GROUPER 50-60 CM','GROUPER >60 CM','GROUPER TOTAL',
						 'MORAY EEL','NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
						 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','NASSAU GROUPER 30-40 CM',
						 'NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
						 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','ORANGE SPOTTED GROUPER 40-50 CM',
					     'ORANGE SPOTTED GROUPER 50-60 CM','ORANGE SPOTTED GROUPER >60 CM',
					     'ORANGE SPOTTED GROUPER TOTAL','PEACOCK GROUPER 40-50 CM',
					     'PEACOCK GROUPER 50-60 CM','PEACOCK GROUPER >60 CM','PEACOCK GROUPER TOTAL',
						 'SHARKS']

############################################
############################################
# Site Decsriptors 
############################################
############################################
SUBSETS.NONSTATIC_DESCRIPTORS = ['Siltation','Dynamite Fishing?','Poison Fishing?',
						 'Aquarium fish collection','Harvest of inverts for food',
						 'Harvest of inverts for curio','Tourist diving/snorkeling',
						 'Sewage pollution','Industrial pollution','Commercial fishing',
						 'Live food fishing','Artisinal/recreational','Other forms of fishing?',
						 'Other Fishing','Yachts','Level of other impacts?','Other impacts?',
						 'Is site protected?','Is protection enforced?','Level of poaching?',
						 'Spearfishing?','Commercial fishing?','Recreational fishing?',
						 'Invertebrate/shell collection?','Anchoring?','Diving?','Other (specify)',
						 'Nature of protection?','Other site description comments?',
						 'Comments from organism sheet','Grouper Size','Percent Bleaching',
						 'Percent colonies bleached','Percent of each colony','Suspected Disease?',
						 'CORAL DAMAGE ANCHOR','CORAL DAMAGE DYNAMITE','CORAL DAMAGE OTHER',
						 'TRASH FISH NETS','TRASH GENERAL','Rare Animals?']

SUBSETS.DIVE_DESCRIPTORS = ['Depth','Time of day work began','Time of day work ended','Weather'
 					'Air Temp','Water temp at surface','Water temp at 5m','Water temp at 10m',
					'Approx popn size (x1000)','Horizontal Visibility in water',
					'Best Reef Area?','Why was this site selected?','Sheltered or exposed?',
					'Any major storms in last years?','When storms?','Overall anthro impact?',
					'What kind of impacts?']


############################################
############################################
# Coral
############################################
############################################
SUBSETS.DISEASES = ['ASPERGILLOSIS','BLACK BAND','WHITE BAND','WHITE PLAGUE']
SUBSETS.CORAL_COVER = ['BLEACHING (% OF COLONY)','BLEACHING (% OF POPULATION)']

############################################
############################################
# Testing Columns 
############################################
############################################
SUBSETS.COMMON_COLUMNS = ['TRASH GENERAL','GROUPER TOTAL','SNAPPER','PENCIL URCHIN','PARROTFISH',
				  'MORAY EEL','LOBSTER','CORAL DAMAGE OTHER','BUTTERFLYFISH']


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv('input/data.csv')

#Extract undefined values:
prediction_dataset = raw_data[raw_data['shot_made_flag'].isnull()]

# Training dataset
test_data = raw_data[raw_data['shot_made_flag'].notnull()]

#Clean non relevant columns
test_data = test_data.drop(['game_event_id', 'game_id', 'team_id', 'team_name', 'shot_id', 'game_date', 'matchup'], axis=1)

#Integer encoding

test_data.action_type=test_data.loc[:,'action_type'].apply(lambda x :  np.where(x == test_data.action_type.unique())[0][0])
test_data.combined_shot_type=test_data.loc[:,'combined_shot_type'].apply(lambda x :  np.where(x == test_data.combined_shot_type.unique())[0][0])
test_data.lat=test_data.loc[:,'lat'].apply(lambda x :  np.where(x == test_data.lat.unique())[0][0])
test_data.loc_x=test_data.loc[:,'loc_x'].apply(lambda x :  np.where(x == test_data.loc_x.unique())[0][0])
test_data.loc_y=test_data.loc[:,'loc_y'].apply(lambda x :  np.where(x == test_data.loc_y.unique())[0][0])
test_data.lon=test_data.loc[:,'lon'].apply(lambda x :  np.where(x == test_data.lon.unique())[0][0])
test_data.minutes_remaining=test_data.loc[:,'minutes_remaining'].apply(lambda x :  np.where(x == test_data.minutes_remaining.unique())[0][0])
test_data.period=test_data.loc[:,'period'].apply(lambda x :  np.where(x == test_data.period.unique())[0][0])
test_data.playoffs=test_data.loc[:,'playoffs'].apply(lambda x :  np.where(x == test_data.playoffs.unique())[0][0])
test_data.season=test_data.loc[:,'season'].apply(lambda x :  np.where(x == test_data.season.unique())[0][0])
test_data.seconds_remaining=test_data.loc[:,'seconds_remaining'].apply(lambda x :  np.where(x == test_data.seconds_remaining.unique())[0][0])
test_data.shot_distance=test_data.loc[:,'shot_distance'].apply(lambda x :  np.where(x == test_data.shot_distance.unique())[0][0])
test_data.shot_type=test_data.loc[:,'shot_type'].apply(lambda x :  np.where(x == test_data.shot_type.unique())[0][0])
test_data.shot_zone_area=test_data.loc[:,'shot_zone_area'].apply(lambda x :  np.where(x == test_data.shot_zone_area.unique())[0][0])
test_data.shot_zone_basic=test_data.loc[:,'shot_zone_basic'].apply(lambda x :  np.where(x == test_data.shot_zone_basic.unique())[0][0])
test_data.shot_zone_range=test_data.loc[:,'shot_zone_range'].apply(lambda x :  np.where(x == test_data.shot_zone_range.unique())[0][0])
test_data.opponent=test_data.loc[:,'opponent'].apply(lambda x :  np.where(x == test_data.opponent.unique())[0][0])




print('Column count after cleanup: ' + str(len(test_data.columns)))
print('Row count after cleanup: ' + str(len(test_data.index)))


print('Test data prepared:')
print(test_data)

#input.to_csv('output/output.csv')
#plt.interactive(False)
#print raw_input.plot(figsize=(30, 30))
#plt.show()

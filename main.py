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


print('Column count after cleanup: ' + str(len(test_data.columns)))
print('Row count after cleanup: ' + str(len(test_data.index)))



#input.to_csv('output/output.csv')
#plt.interactive(False)
#print raw_input.plot(figsize=(30, 30))
#plt.show()

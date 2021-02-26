import pandas as pd
import numpy as np
import json


def filterBySeason(df, year, season):
    """
    This method allows to filter by the seasons that the company has to evaluate its metrics
    """
    if season == 'spring':
        return df[(df['start_date'] >= str(year) + "-01-01") & (df['start_date'] < str(year) + "-03-01")]
    elif season == 'summer':
        return df[(df['start_date'] >= str(year) + "-03-01") & (df['start_date'] < str(year) + "-09-01")]
    elif season == 'fall':
        return df[(df['start_date'] >= str(year) + "-09-01") & (df['start_date'] < str(year) + "-11-16")]
    elif season == 'winter':
        return df[(df['start_date'] >= str(year) + "-11-16") & (df['start_date'] < str(year+1) + "-01-01")]
    else:
        raise(Exception('Not valid season'))


def addSeasonColumn(df):
    """
    This method add the respective season to a new dataframe
    @return DataFrame
    """

    df_with_seasons = df.copy(deep=True)
    df_with_seasons['season'] = np.nan
    years = list(
        set(np.array(pd.DatetimeIndex(df_with_seasons['start_date']).year)))
    seasons = ['spring', 'summer', 'fall', 'winter']
    for year in years:
        for season in seasons:
            filterSeasion = filterBySeason(df_with_seasons, year, season)
            if(len(filterSeasion) > 0):
                df_with_seasons.loc[filterSeasion.index, 'season'] = season
    return df_with_seasons


def generateDictSeasonalityUnormalization(df, path="tools/unormalizeSeason.txt", normalizationType='standard'):
    dictSeason = {'spring': {}, 'summer': {}, 'fall': {}, 'winter': {}}
    df_unormalize = df.copy(deep=True)
    if('season' not in df_unormalize.columns):
        df_unormalize = addSeasonColumn(df)

    for season in dictSeason.keys():
        filterSeason = df_unormalize[df_unormalize['season'] == season]
        max_stad = filterSeason['appts'].max()
        min_stad = filterSeason['appts'].min()
        mean = filterSeason['appts'].mean()
        std = filterSeason['appts'].std()
        if(normalizationType == 'standard'):
            dictSeason[season]['mean'] = mean
            dictSeason[season]['std'] = std
        elif(normalizationType == 'maxmin'):
            dictSeason[season]['max'] = max_stad
            dictSeason[season]['min'] = min_stad
        elif(normalizationType == 'mean'):
            dictSeason[season]['mean'] = mean
            dictSeason[season]['max'] = max_stad
            dictSeason[season]['min'] = min_stad
        elif(normalizationType == 'submean'):
            dictSeason[season]['mean'] = mean
        else:
            raise(Exception("Tipo de normalizacion incorrecta"))

    dictSeason['normalizationType'] = normalizationType
    unormalizationFile = open(path, 'w')
    unormalizationFile.write(json.dumps(dictSeason))
    return dictSeason


def readUnormalizationFile(path="unormalizeSeason.txt"):
    return json.loads(open(path, 'r').read())


def normaliceSeason(df, year, season, normalizationType):
    filterSeason = filterBySeason(df, year, season)
    if(len(filterSeason) > 0):
        mean = filterSeason['appts'].mean()
        std = filterSeason['appts'].std()
        max_stad = filterSeason['appts'].max()
        min_stad = filterSeason['appts'].min()

        normFunctions = {'standard': lambda x: (
            x-mean)/std, 'maxmin': lambda x: (x-min_stad)/(max_stad-min_stad), 'mean': lambda x: (x-mean)/(max_stad-min_stad),'submean':lambda x: (x-mean)}
        indices = filterSeason.index
        df.loc[indices, 'appts'] = df.loc[indices, 'appts'].apply(
            normFunctions[normalizationType])
    return df


def normalizationBySeasonality(df, normalizationType='standard'):
    df_normalizado = df.copy(deep=True)
    years = list(
        set(np.array(pd.DatetimeIndex(df_normalizado['start_date']).year)))
    seasons = ['spring', 'summer', 'fall', 'winter']
    generateDictSeasonalityUnormalization(
        df_normalizado, normalizationType=normalizationType)
    for year in years:
        for season in seasons:
            normaliceSeason(df_normalizado, year, season, normalizationType)
    
    return df_normalizado

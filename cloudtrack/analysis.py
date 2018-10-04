import pandas as pd
import numpy as np
import logging
import os

from .utils import mask_cell,mask_cell_surface,mask_cube_cell,get_bounding_box

def cell_statistics_all(input_cubes,track,mask,aggregators,output_path='./',cell_selection=None,output_name='Profiles',width=10000,z_coord='model_level_number',dimensions=['x','y'],**kwargs):
    if cell_selection is None:
        cell_selection=np.unique(track['cell'])
    for cell in cell_selection :
        cell_statistics(input_cubes=input_cubes,track=track, mask=mask,
                        dimensions=dimensions,aggregators=aggregators,cell=cell,
                        output_path=output_path,output_name=output_name,
                        width=width,z_coord=z_coord,**kwargs)

def cell_statistics(input_cubes,track,mask,aggregators,cell,output_path='./',output_name='Profiles',width=10000,z_coord='model_level_number',dimensions=['x','y'],**kwargs):
    from iris.cube import Cube,CubeList
    from iris.coords import AuxCoord
    from iris import Constraint,save    
    
    # If input is single cube, turn into cubelist
    if type(input_cubes) is Cube:
        input_cubes=CubeList([input_cubes])

    
    logging.debug('Start calculating profiles for cell '+str(cell))
    track_i=track[track['cell']==cell]
    
    cubes_profile={}
    for aggregator in aggregators:
        cubes_profile[aggregator.name()]=CubeList()
        
    for time_i in track_i['time'].values:
        constraint_time = Constraint(time=time_i)
        
        mask_i=mask.extract(constraint_time)
        mask_cell_i=mask_cell(mask_i,track_i,cell,masked=False)
        mask_cell_surface_i=mask_cell_surface(mask_i,track_i,cell,masked=False,z_coord=z_coord)

        x_dim=mask_cell_surface_i.coord_dims('projection_x_coordinate')[0]
        y_dim=mask_cell_surface_i.coord_dims('projection_y_coordinate')[0]
        x_coord=mask_cell_surface_i.coord('projection_x_coordinate')
        y_coord=mask_cell_surface_i.coord('projection_y_coordinate')
    
        if (mask_cell_surface_i.core_data()>0).any():
            box_mask_i=get_bounding_box(mask_cell_surface_i.core_data(),buffer=1)
    
            box_mask=[[x_coord.points[box_mask_i[x_dim][0]],x_coord.points[box_mask_i[x_dim][1]]],
                     [y_coord.points[box_mask_i[y_dim][0]],y_coord.points[box_mask_i[y_dim][1]]]]
        else:
            box_mask=[[np.nan,np.nan],[np.nan,np.nan]]
    
        x=track_i[track_i['time'].values==time_i]['projection_x_coordinate'].values[0]
        y=track_i[track_i['time'].values==time_i]['projection_y_coordinate'].values[0]

        box_slice=[[x-width,x+width],[y-width,y+width]]
               
        x_min=np.nanmin([box_mask[0][0],box_slice[0][0]])
        x_max=np.nanmax([box_mask[0][1],box_slice[0][1]])
        y_min=np.nanmin([box_mask[1][0],box_slice[1][0]])
        y_max=np.nanmax([box_mask[1][1],box_slice[1][1]])

        constraint_x=Constraint(projection_x_coordinate=lambda cell: int(x_min) < cell < int(x_max))
        constraint_y=Constraint(projection_y_coordinate=lambda cell: int(y_min) < cell < int(y_max))

        constraint=constraint_time & constraint_x & constraint_y
#       Mask_cell_surface_i=mask_cell_surface(Mask_w_i,cell,masked=False,z_coord='model_level_number')
        mask_cell_i=mask_cell_i.extract(constraint)
        mask_cell_surface_i=mask_cell_surface_i.extract(constraint)

        input_cubes_i=input_cubes.extract(constraint)
        for cube in input_cubes_i:
            cube_masked=mask_cube_cell(cube,mask_cell_i,cell)
            for aggregator in aggregators:
                cubes_profile[aggregator.name()].append(cube_masked.collapsed(dimensions,aggregator,**kwargs))


    minutes=(track_i['time_cell']/pd.Timedelta(minutes=1)).values
    latitude=track_i['latitude'].values
    longitude=track_i['longitude'].values
    minutes_coord=AuxCoord(minutes,long_name='cell_time',units='min')
    latitude_coord=AuxCoord(latitude,long_name='latitude',units='degrees')
    longitude_coord=AuxCoord(longitude,long_name='longitude',units='degrees')
    
    for aggregator in aggregators:
        
        cubes_profile[aggregator.name()]=cubes_profile[aggregator.name()].merge()
        for cube in cubes_profile[aggregator.name()]:
            for coord in  cube.coords():
                if (coord.ndim>1 and (cube.coord_dims(dimensions[0])[0] in cube.coord_dims(coord) or cube.coord_dims(dimensions[1])[0] in cube.coord_dims(coord))):
                    cube.remove_coord(coord.name())
                
            cube.add_aux_coord(minutes_coord,data_dims=cube.coord_dims('time'))
            cube.add_aux_coord(latitude_coord,data_dims=cube.coord_dims('time'))
            cube.add_aux_coord(longitude_coord,data_dims=cube.coord_dims('time'))
        
        os.makedirs(os.path.join(output_path,output_name,aggregator.name()),exist_ok=True)
        savefile=os.path.join(output_path,output_name,aggregator.name(),output_name+'_'+ aggregator.name()+'_'+str(int(cell))+'.nc')
        save(cubes_profile[aggregator.name()],savefile)


def cog_cell(cell,Tracks=None,M_total=None,M_liquid=None,
             M_frozen=None,
             Mask=None,
             savedir=None):
    
    
    from iris import Constraint
    logging.debug('Start calculating COG for '+str(cell))
    Track=Tracks[Tracks['cell']==cell]
    constraint_time=Constraint(time=lambda cell: Track.head(1)['time'].values[0] <= cell <= Track.tail(1)['time'].values[0])
    M_total_i=M_total.extract(constraint_time)
    M_liquid_i=M_liquid.extract(constraint_time)
    M_frozen_i=M_frozen.extract(constraint_time)
    Mask_i=Mask.extract(constraint_time)

    savedir_cell=os.path.join(savedir,'cells',str(int(cell)))
    os.makedirs(savedir_cell,exist_ok=True)
    savefile_COG_total_i=os.path.join(savedir_cell,'COG_total'+'_'+str(int(cell))+'.h5')
    savefile_COG_liquid_i=os.path.join(savedir_cell,'COG_liquid'+'_'+str(int(cell))+'.h5')
    savefile_COG_frozen_i=os.path.join(savedir_cell,'COG_frozen'+'_'+str(int(cell))+'.h5')
    
    Tracks_COG_total_i=calculate_cog(Track,M_total_i,Mask_i)
#   Tracks_COG_total_list.append(Tracks_COG_total_i)
    logging.debug('COG total loaded for ' +str(cell))
    
    Tracks_COG_liquid_i=calculate_cog(Track,M_liquid_i,Mask_i)
#   Tracks_COG_liquid_list.append(Tracks_COG_liquid_i)
    logging.debug('COG liquid loaded for ' +str(cell))
    Tracks_COG_frozen_i=calculate_cog(Track,M_frozen_i,Mask_i)
#   Tracks_COG_frozen_list.append(Tracks_COG_frozen_i)
    logging.debug('COG frozen loaded for ' +str(cell))
    
    Tracks_COG_total_i.to_hdf(savefile_COG_total_i,'table')
    Tracks_COG_liquid_i.to_hdf(savefile_COG_liquid_i,'table')
    Tracks_COG_frozen_i.to_hdf(savefile_COG_frozen_i,'table')
    logging.debug('individual COG calculated and saved to '+ savedir_cell)


def lifetime_histogram(Track,bin_edges=np.arange(0,200,20),density=False):
    Track.at['v']=np.nan
    Track_cell=Track.groupby('cell')
    minutes=(Track_cell['time_cell'].max()/pd.Timedelta(minutes=1)).values
    hist, bin_edges = np.histogram(minutes, bin_edges,density=density)
    return hist,bin_edges

def calculate_distance(feature_1,feature_2):
    distance=np.sqrt((feature_1['projection_x_coordinate']-feature_2['projection_x_coordinate'])**2
                     +(feature_1['projection_y_coordinate']-feature_2['projection_y_coordinate'])**2)
    return distance

def calculate_velocity(feature_old,feature_new):
    distance=calculate_distance(feature_old,feature_new)
    diff_time=((feature_new['time']-feature_old['time']).total_seconds())
    velocity=distance/diff_time
    return velocity

def velocity_histogram(Track,bin_edges=np.arange(0,30,1),density=False):
    for cell_i,Track_i in Track.groupby('cell'):
        index=Track_i.index.values
        for i,index_i in enumerate(index[:-1]):
            velocity=calculate_velocity(Track_i.loc[index[i]],Track_i.loc[index[i+1]])
            Track.at[index_i,'v']=velocity
    velocities=Track['v'].values
    hist, bin_edges = np.histogram(velocities[~np.isnan(velocities)], bin_edges,density=density)
    return hist,bin_edges

def nearestneighbordistance_histogram(features,bin_edges=np.arange(0,30000,500),density=False):
    from itertools import combinations
    features['min_distance']=np.nan
    for time_i,features_i in features.groupby('time'):
            indeces=combinations(features_i.index.values,2)
            #Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
            distances=[]
            for index_1,index_2 in indeces:        
                if index_1 is not index_2:                    
                    distance=calculate_distance(features_i.loc[index_1],features_i.loc[index_2])
                    distances.append(pd.DataFrame({'index_1':index_1,'index_2':index_2,'distance': distance}, index=[0]))
            if any([x is not None for x in distances]):
                distances=pd.concat(distances, ignore_index=True)   
                for i in features_i.index:
                    min_distance=distances.loc[(distances['index_1']==i) | (distances['index_2']==i),'distance'].min()
                    features.at[i,'min_distance']=min_distance
    distances=features['min_distance'].values
    hist, bin_edges = np.histogram(distances[~np.isnan(distances)], bin_edges,density=density)
    return hist,bin_edges

def histogram_cellwise(Track,variable=None,bin_edges=None,quantity='max',density=False):
    Track_cell=Track.groupby('cell')
    if quantity=='max':
        variable_cell=Track_cell[variable].max().values
    elif quantity=='min':
        variable_cell=Track_cell[variable].min().values
    elif quantity=='mean':
        variable_cell=Track_cell[variable].mean().values
    else:
        raise ValueError('quantity unknown, must be max, min or mean')
    hist, bin_edges = np.histogram(variable_cell, bin_edges,density=density)
    return hist,bin_edges

def histogram_featurewise(Track,variable=None,bin_edges=None,density=False):
    hist, bin_edges = np.histogram(Track[variable].values, bin_edges,density=density)
    return hist,bin_edges

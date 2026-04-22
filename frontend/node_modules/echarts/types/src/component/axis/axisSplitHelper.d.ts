import * as graphic from '../../util/graphic.js';
import GridModel from '../../coord/cartesian/GridModel.js';
import type SingleAxisView from './SingleAxisView.js';
import type CartesianAxisView from './CartesianAxisView.js';
import type SingleAxisModel from '../../coord/single/AxisModel.js';
import type CartesianAxisModel from '../../coord/cartesian/AxisModel.js';
export declare function rectCoordAxisBuildSplitArea(axisView: SingleAxisView | CartesianAxisView, axisGroup: graphic.Group, axisModel: SingleAxisModel | CartesianAxisModel, gridModel: GridModel | SingleAxisModel): void;
export declare function rectCoordAxisHandleRemove(axisView: SingleAxisView | CartesianAxisView): void;

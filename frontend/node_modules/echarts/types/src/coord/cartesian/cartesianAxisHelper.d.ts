import GridModel from './GridModel.js';
import CartesianAxisModel from './AxisModel.js';
import SeriesModel from '../../model/Series.js';
interface CartesianAxisLayout {
    position: [number, number];
    rotation: number;
    labelOffset: number;
    labelDirection: -1 | 1;
    tickDirection: -1 | 1;
    nameDirection: -1 | 1;
    labelRotate: number;
    z2: number;
}
/**
 * Can only be called after coordinate system creation stage.
 * (Can be called before coordinate system update stage).
 */
export declare function layout(gridModel: GridModel, axisModel: CartesianAxisModel, opt?: {
    labelInside?: boolean;
}): CartesianAxisLayout;
export declare function isCartesian2DSeries(seriesModel: SeriesModel): boolean;
export declare function findAxisModels(seriesModel: SeriesModel): {
    xAxisModel: CartesianAxisModel;
    yAxisModel: CartesianAxisModel;
};
export {};

import * as graphic from '../../util/graphic.js';
import SeriesModel from '../../model/Series.js';
import { SeriesOption } from '../../util/types.js';
import type Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
import type Polar from '../../coord/polar/Polar.js';
import { CoordinateSystem } from '../../coord/CoordinateSystem.js';
declare type SeriesModelWithLineWidth = SeriesModel<SeriesOption & {
    lineStyle?: {
        width?: number;
    };
}>;
declare function createGridClipPath(cartesian: Cartesian2D, hasAnimation: boolean, seriesModel: SeriesModelWithLineWidth, done?: () => void, during?: (percent: number, clipRect: graphic.Rect) => void): graphic.Rect;
declare function createPolarClipPath(polar: Polar, hasAnimation: boolean, seriesModel: SeriesModelWithLineWidth): graphic.Sector;
declare function createClipPath(coordSys: CoordinateSystem, hasAnimation: boolean, seriesModel: SeriesModelWithLineWidth, done?: () => void, during?: (percent: number) => void): graphic.Rect | graphic.Sector;
export { createGridClipPath, createPolarClipPath, createClipPath };

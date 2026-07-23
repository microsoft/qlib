import SeriesData from '../../data/SeriesData.js';
import type Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
import type Polar from '../../coord/polar/Polar.js';
import { LineSeriesOption } from './LineSeries.js';
interface DiffItem {
    cmd: '+' | '=' | '-';
    idx: number;
    idx1?: number;
}
export default function lineAnimationDiff(oldData: SeriesData, newData: SeriesData, oldStackedOnPoints: ArrayLike<number>, newStackedOnPoints: ArrayLike<number>, oldCoordSys: Cartesian2D | Polar, newCoordSys: Cartesian2D | Polar, oldValueOrigin: LineSeriesOption['areaStyle']['origin'], newValueOrigin: LineSeriesOption['areaStyle']['origin']): {
    current: number[] | Float32Array;
    next: number[] | Float32Array;
    stackedOnCurrent: number[] | Float32Array;
    stackedOnNext: number[] | Float32Array;
    status: DiffItem[];
};
export {};

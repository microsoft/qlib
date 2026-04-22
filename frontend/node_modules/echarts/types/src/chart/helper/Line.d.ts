import * as graphic from '../../util/graphic.js';
import SeriesData from '../../data/SeriesData.js';
import SeriesModel from '../../model/Series.js';
import type { LineDrawSeriesScope } from './LineDraw.js';
import { LineDataVisual } from '../../visual/commonVisualTypes.js';
declare type LineList = SeriesData<SeriesModel, LineDataVisual>;
export interface LineLabel extends graphic.Text {
    lineLabelOriginalOpacity: number;
}
declare class Line extends graphic.Group {
    private _fromSymbolType;
    private _toSymbolType;
    constructor(lineData: SeriesData, idx: number, seriesScope?: LineDrawSeriesScope);
    _createLine(lineData: LineList, idx: number, seriesScope?: LineDrawSeriesScope): void;
    updateData(lineData: SeriesData, idx: number, seriesScope: LineDrawSeriesScope): void;
    getLinePath(): graphic.Line;
    _updateCommonStl(lineData: SeriesData, idx: number, seriesScope?: LineDrawSeriesScope): void;
    highlight(): void;
    downplay(): void;
    updateLayout(lineData: SeriesData, idx: number): void;
    setLinePoints(points: number[][]): void;
    beforeUpdate(): void;
}
export default Line;

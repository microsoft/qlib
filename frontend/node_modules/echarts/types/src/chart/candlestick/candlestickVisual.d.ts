import { StageHandler } from '../../util/types.js';
import { CandlestickDataItemOption } from './CandlestickSeries.js';
import Model from '../../model/Model.js';
export declare function getColor(sign: number, model: Model<Pick<CandlestickDataItemOption, 'itemStyle'>>): import("../../util/types").ZRColor;
export declare function getBorderColor(sign: number, model: Model<Pick<CandlestickDataItemOption, 'itemStyle'>>): import("../../util/types").ZRColor;
declare const candlestickVisual: StageHandler;
export default candlestickVisual;

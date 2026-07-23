import { StageHandler } from '../../util/types.js';
import { RectLike } from 'zrender/lib/core/BoundingRect.js';
export interface CandlestickItemLayout {
    sign: number;
    initBaseline: number;
    ends: number[][];
    brushRect: RectLike;
}
export interface CandlestickLayoutMeta {
    candleWidth: number;
    isSimpleBox: boolean;
}
declare const candlestickLayout: StageHandler;
export default candlestickLayout;

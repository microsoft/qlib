/**
 * A mapping of visual provided to developer and visual stored in the List module.
 * To developer:
 *  'color', 'opacity', 'symbol', 'symbolSize'...
 * In the List module storage:
 *  'style', 'symbol', 'symbolSize'...
 */
import SeriesData from '../data/SeriesData.js';
export declare function getItemVisualFromData(data: SeriesData, dataIndex: number, key: string): string | number | number[] | import("zrender/lib/graphic/Pattern").PatternObject | import("zrender/lib/graphic/LinearGradient").LinearGradientObject | import("zrender/lib/graphic/RadialGradient").RadialGradientObject;
export declare function getVisualFromData(data: SeriesData, key: string): string | number | number[] | import("zrender/lib/graphic/Pattern").PatternObject | import("zrender/lib/graphic/LinearGradient").LinearGradientObject | import("zrender/lib/graphic/RadialGradient").RadialGradientObject;
export declare function setItemVisualFromData(data: SeriesData, dataIndex: number, key: string, value: any): void;

import SeriesModel from '../../model/Series.js';
import { TooltipMarkupSection } from './tooltipMarkup.js';
export declare function defaultSeriesFormatTooltip(opt: {
    series: SeriesModel;
    dataIndex: number;
    multipleSeries: boolean;
}): TooltipMarkupSection;

import BoundingRect, { RectLike } from 'zrender/lib/core/BoundingRect.js';
import CalendarModel from './CalendarModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { LayoutOrient, ScaleDataValue, OptionDataValueDate } from '../../util/types.js';
import { ParsedModelFinder } from '../../util/model.js';
import { CoordinateSystem, CoordinateSystemMaster } from '../CoordinateSystem.js';
export interface CalendarParsedDateRangeInfo {
    range: [string, string];
    start: CalendarParsedDateInfo;
    end: CalendarParsedDateInfo;
    allDay: number;
    weeks: number;
    nthWeek: number;
    fweek: number;
    lweek: number;
}
export interface CalendarParsedDateInfo {
    /**
     * local full year, eg., '1940'
     */
    y: string;
    /**
     * local month, from '01' ot '12',
     */
    m: string;
    /**
     * local date, from '01' to '31' (if exists),
     */
    d: string;
    /**
     * It is not date.getDay(). It is the location of the cell in a week, from 0 to 6,
     */
    day: number;
    /**
     * Timestamp
     */
    time: number;
    /**
     * yyyy-MM-dd
     */
    formatedDate: string;
    /**
     * The original date object
     */
    date: Date;
}
export interface CalendarCellRect {
    contentShape: RectLike;
    center: number[];
    tl: number[];
    tr: number[];
    br: number[];
    bl: number[];
}
declare class Calendar implements CoordinateSystem, CoordinateSystemMaster {
    static readonly dimensions: string[];
    static getDimensionsInfo(): (string | {
        name: string;
        type: "time";
    })[];
    readonly type = "calendar";
    readonly dimensions: string[];
    private _model;
    private _rect;
    private _sw;
    private _sh;
    private _orient;
    private _firstDayOfWeek;
    private _rangeInfo;
    private _lineWidth;
    constructor(calendarModel: CalendarModel, ecModel: GlobalModel, api: ExtensionAPI);
    getDimensionsInfo: typeof Calendar.getDimensionsInfo;
    getRangeInfo(): CalendarParsedDateRangeInfo;
    getModel(): CalendarModel;
    getRect(): BoundingRect;
    getCellWidth(): number;
    getCellHeight(): number;
    getOrient(): LayoutOrient;
    /**
     * getFirstDayOfWeek
     *
     * @example
     *     0 : start at Sunday
     *     1 : start at Monday
     *
     * @return {number}
     */
    getFirstDayOfWeek(): number;
    /**
     * get date info
     * }
     */
    getDateInfo(date: OptionDataValueDate): CalendarParsedDateInfo;
    getNextNDay(date: OptionDataValueDate, n: number): CalendarParsedDateInfo;
    update(ecModel: GlobalModel, api: ExtensionAPI): void;
    /**
     * Convert a time data(time, value) item to (x, y) point.
     */
    dataToPoint(data: OptionDataValueDate | OptionDataValueDate[], clamp?: boolean): number[];
    /**
     * Convert a (x, y) point to time data
     */
    pointToData(point: number[]): number;
    /**
     * Convert a time date item to (x, y) four point.
     */
    dataToRect(data: OptionDataValueDate | OptionDataValueDate[], clamp?: boolean): CalendarCellRect;
    /**
     * Convert a (x, y) point to time date
     *
     * @param  {Array} point point
     * @return {Object}       date
     */
    pointToDate(point: number[]): CalendarParsedDateInfo;
    convertToPixel(ecModel: GlobalModel, finder: ParsedModelFinder, value: ScaleDataValue | ScaleDataValue[]): number[];
    convertFromPixel(ecModel: GlobalModel, finder: ParsedModelFinder, pixel: number[]): number;
    containPoint(point: number[]): boolean;
    /**
     * initRange
     * Normalize to an [start, end] array
     */
    private _initRangeOption;
    /**
     * range info
     *
     * @private
     * @param  {Array} range range ['2017-01-01', '2017-07-08']
     *  If range[0] > range[1], they will not be reversed.
     * @return {Object}       obj
     */
    _getRangeInfo(range: OptionDataValueDate[]): CalendarParsedDateRangeInfo;
    /**
     * get date by nthWeeks and week day in range
     *
     * @private
     * @param  {number} nthWeek the week
     * @param  {number} day   the week day
     * @param  {Array} range [d1, d2]
     * @return {Object}
     */
    private _getDateByWeeksAndDay;
    static create(ecModel: GlobalModel, api: ExtensionAPI): Calendar[];
}
export default Calendar;

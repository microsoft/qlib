declare type SSRItemType = 'legend' | 'chart';
export interface ECSSRClientEventParams {
    type: ECSSREvent;
    ssrType: SSRItemType;
    seriesIndex?: number;
    dataIndex?: number;
    event: Event;
}
export interface ECSSRClientOptions {
    on?: {
        mouseover?: (params: ECSSRClientEventParams) => void;
        mouseout?: (params: ECSSRClientEventParams) => void;
        click?: (params: ECSSRClientEventParams) => void;
    };
}
export declare type ECSSREvent = 'mouseover' | 'mouseout' | 'click';
export declare function hydrate(dom: HTMLElement, options: ECSSRClientOptions): void;
export {};

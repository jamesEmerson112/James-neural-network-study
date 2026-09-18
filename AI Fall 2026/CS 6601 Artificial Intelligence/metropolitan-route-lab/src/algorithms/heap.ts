export interface HeapItem {
  node: number;
  g: number;
  key: number;
}

export class MinHeap {
  private data: HeapItem[] = [];

  get size(): number {
    return this.data.length;
  }

  peek(): HeapItem | undefined {
    return this.data[0];
  }

  push(item: HeapItem): void {
    const data = this.data;
    data.push(item);
    let index = data.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (data[parent].key <= item.key) break;
      data[index] = data[parent];
      index = parent;
    }
    data[index] = item;
  }

  pop(): HeapItem | undefined {
    const data = this.data;
    if (!data.length) return undefined;
    const root = data[0];
    const tail = data.pop();
    if (data.length && tail) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        if (left >= data.length) break;
        const right = left + 1;
        const child = right < data.length && data[right].key < data[left].key ? right : left;
        if (data[child].key >= tail.key) break;
        data[index] = data[child];
        index = child;
      }
      data[index] = tail;
    }
    return root;
  }
}

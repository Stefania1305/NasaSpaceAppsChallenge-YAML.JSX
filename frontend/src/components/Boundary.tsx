import { Component, type ReactNode } from 'react';

// Keeps one failing component (e.g. a server answer with an unexpected shape) from blanking the whole page.
// `resetKey`: when it changes (a new run, say), the children are tried again.
export class Boundary extends Component<{ children: ReactNode; fallback: ReactNode; resetKey?: unknown }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(err: unknown) { console.error(err); }
  componentDidUpdate(prev: { resetKey?: unknown }) {
    if (this.state.failed && prev.resetKey !== this.props.resetKey) this.setState({ failed: false });
  }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

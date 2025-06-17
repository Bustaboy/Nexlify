// Location: /src/components/common/TimeRangeSelector/index.tsx
// Time range selector for chart displays

import React, { useState } from 'react';
import styled from 'styled-components';

export type TimeRange = '1h' | '1d' | '7d' | '30d';

interface TimeRangeSelectorProps {
  defaultValue?: TimeRange;
  onChange?: (range: TimeRange) => void;
  className?: string;
}

const Container = styled.div`
  display: flex;
  border: 2px solid var(--color-primary-60);
  border-radius: var(--radius-lg);
  overflow: hidden;
  backdrop-filter: blur(8px);
`;

const RangeButton = styled.button<{ $active: boolean }>`
  padding: 0.375rem 0.75rem;
  font-size: 0.75rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  background: ${({ $active }) => 
    $active ? 'var(--color-primary)' : 'rgba(31, 41, 55, 0.5)'
  };
  color: ${({ $active }) => $active ? 'white' : '#9CA3AF'};
  border: none;
  cursor: pointer;
  transition: all var(--transition-fast);
  position: relative;
  
  &:not(:last-child) {
    border-right: 1px solid var(--color-primary-40);
  }
  
  &:hover:not(:disabled) {
    background: ${({ $active }) => 
      $active ? 'var(--color-primary)' : 'rgba(31, 41, 55, 0.8)'
    };
    color: ${({ $active }) => $active ? 'white' : '#D1D5DB'};
  }
  
  &:focus-visible {
    outline: none;
    z-index: 1;
    box-shadow: inset 0 0 0 2px var(--color-primary);
  }
  
  ${({ $active }) => $active && `
    box-shadow: 0 0 15px var(--color-primary-80);
  `}
`;

const TIME_RANGES: TimeRange[] = ['1h', '1d', '7d', '30d'];

export const TimeRangeSelector: React.FC<TimeRangeSelectorProps> = ({
  defaultValue = '1d',
  onChange,
  className = ''
}) => {
  const [selected, setSelected] = useState<TimeRange>(defaultValue);
  
  const handleSelect = (range: TimeRange) => {
    setSelected(range);
    onChange?.(range);
  };
  
  return (
    <Container className={className}>
      {TIME_RANGES.map(range => (
        <RangeButton
          key={range}
          $active={selected === range}
          onClick={() => handleSelect(range)}
          aria-pressed={selected === range}
        >
          {range}
        </RangeButton>
      ))}
    </Container>
  );
};

export default TimeRangeSelector;

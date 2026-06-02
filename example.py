from sequence import RequestStatus, SamplingParams, Sequence

## creating a req object
req = RequestStatus.WAITING
print(req)

## creating a sampling params object
sam = SamplingParams()
seq = Sequence(request_id = 1, sampling_params=sam, prompt_token_ids=[1, 2, 3])
print(seq)